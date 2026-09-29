import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3, efficientnet_b4
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = []

ARCHS = {"efficientnet_b3": efficientnet_b3, "efficientnet_b4": efficientnet_b4}


def parse_args():
    p = argparse.ArgumentParser(
        description="Detect, track and identify JDM cars in a video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", required=True, help="input video path")
    p.add_argument("-o", "--output", default=None,
                   help="output video path (default: <input>_annotated.mp4)")

    g = p.add_argument_group("classifier")
    g.add_argument("--b3", default=os.path.join(BASE_DIR, "best_b3.pth"),
                   help="EfficientNet-B3 checkpoint")
    g.add_argument("--b4", default=os.path.join(BASE_DIR, "best_b4.pth"),
                   help="EfficientNet-B4 checkpoint")
    g.add_argument("--b3-weight", type=float, default=0.5, help="ensemble weight for B3")
    g.add_argument("--b4-weight", type=float, default=0.5, help="ensemble weight for B4")
    g.add_argument("--temperature", type=float, default=1.0, help="softmax temperature")
    g.add_argument("--img-size", type=int, default=224, help="classifier input size")
    g.add_argument("--class-names", default=None,
                   help="optional JSON file with a list of class names (overrides built-in list)")
    g.add_argument("--min-conf", type=float, default=0.35,
                   help="labels below this confidence are shown with a '?' prefix")

    g = p.add_argument_group("detection / tracking")
    g.add_argument("--yolo", default="yolov8n.pt", help="YOLO weights")
    g.add_argument("--classes", type=int, nargs="+", default=[2],
                   help="COCO class ids to treat as cars (2=car, 5=bus, 7=truck)")
    g.add_argument("--det-conf", type=float, default=0.4, help="YOLO confidence threshold")
    g.add_argument("--min-crop", type=int, default=32,
                   help="skip classification for crops smaller than this (pixels)")
    g.add_argument("--pad", type=float, default=0.0,
                   help="expand each crop by this fraction of box size before classifying")

    g = p.add_argument_group("performance")
    g.add_argument("--classify-every", type=int, default=30,
                   help="re-classify each track every N frames (new tracks are classified at once)")
    g.add_argument("--device", default="auto", help="auto | cpu | cuda")
    g.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")

    g = p.add_argument_group("output")
    g.add_argument("--csv", default=None, help="write per-track summary to this CSV")
    g.add_argument("--show", action="store_true", help="preview window (press q to quit)")
    return p.parse_args()


# Classifier
def load_backbone(arch, path, class_names, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    print(f"Loading {arch} from {path}")
    num_classes = len(class_names)

    model = ARCHS[arch](weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Full training checkpoints (optimizer state etc.) can need the permissive loader.
        # Only do this for files you trust.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    # Strip "module." prefix if the checkpoint came from a DDP-wrapped model
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}

    head = state.get("classifier.1.weight")
    if head is not None and head.shape[0] != num_classes:
        raise RuntimeError(
            f"{path} has {head.shape[0]} output classes but the class list has {num_classes}. "
            "Fix CLASS_NAMES / --class-names so it matches training."
        )

    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


class EnsembleClassifier:

    def __init__(self, args, class_names, device):
        self.device = device
        self.class_names = class_names
        self.temperature = args.temperature

        total = args.b3_weight + args.b4_weight
        if total <= 0:
            raise ValueError("Ensemble weights must sum to a positive number")
        self.w3 = args.b3_weight / total
        self.w4 = args.b4_weight / total

        self.b3 = load_backbone("efficientnet_b3", args.b3, class_names, device)
        self.b4 = load_backbone("efficientnet_b4", args.b4, class_names, device)

        self.transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def predict_batch(self, crops_bgr):
        tensors = [
            self.transform(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
            for c in crops_bgr
        ]
        batch = torch.stack(tensors).to(self.device)

        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, enabled=use_amp):
            p3 = F.softmax(self.b3(batch).float() / self.temperature, dim=1)
            p4 = F.softmax(self.b4(batch).float() / self.temperature, dim=1)

        return (self.w3 * p3 + self.w4 * p4).cpu().numpy()


class TrackState:
    def __init__(self, num_classes):
        self.prob_sum = np.zeros(num_classes, dtype=np.float64)
        self.n_cls = 0
        self.frames_seen = 0
        self.last_cls_frame = -10**9

    def update(self, probs, frame_idx):
        self.prob_sum += probs
        self.n_cls += 1
        self.last_cls_frame = frame_idx

    def best(self):
        if self.n_cls == 0:
            return None, 0.0
        mean = self.prob_sum / self.n_cls
        idx = int(mean.argmax())
        return idx, float(mean[idx])


def track_color(track_id):
    if track_id is None:
        return (0, 255, 0)
    rng = np.random.default_rng(track_id * 7919)
    return tuple(int(v) for v in rng.integers(60, 255, size=3))


def draw_label(frame, x1, y1, text, color):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    top = max(0, y1 - th - base - 6)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + base + 6), color, -1)
    cv2.putText(frame, text, (x1 + 3, top + th + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def open_writer(path, fps, size):
    root, ext = os.path.splitext(path)
    attempts = [(path, "mp4v")]
    if ext.lower() != ".avi":
        attempts.append((root + ".avi", "XVID"))
    else:
        attempts[0] = (path, "XVID")

    for out_path, codec in attempts:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            print(f"Output: {os.path.abspath(out_path)} (codec {codec})")
            return writer, out_path
        writer.release()
    raise RuntimeError("Could not open a VideoWriter with mp4v or XVID codecs")


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return torch.device(name)


def print_summary(tracks, class_names, csv_path):
    rows = []
    for tid, st in tracks.items():
        idx, conf = st.best()
        rows.append({
            "track_id": tid,
            "label": class_names[idx] if idx is not None else "",
            "confidence": round(conf, 4),
            "frames_seen": st.frames_seen,
            "classifications": st.n_cls,
        })
    rows.sort(key=lambda r: r["frames_seen"], reverse=True)

    print("\nTrack summary (top 20 by frames seen)")
    print(f"{'ID':>5}  {'Label':<26}{'Conf':>6}{'Frames':>8}{'Cls':>5}")
    for r in rows[:20]:
        print(f"{r['track_id']:>5}  {r['label']:<26}{r['confidence']:>6.2f}"
              f"{r['frames_seen']:>8}{r['classifications']:>5}")

    if csv_path and rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSummary CSV: {os.path.abspath(csv_path)}")


def run(args):
    class_names = CLASS_NAMES
    if args.class_names:
        with open(args.class_names, "r", encoding="utf-8") as f:
            class_names = json.load(f)

    device = resolve_device(args.device)
    print(f"Device: {device}")

    print("Loading YOLO...")
    detector = YOLO(args.yolo)
    classifier = EnsembleClassifier(args, class_names, device)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames > 0 and total_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    print(f"\nInput: {args.input}\nFPS: {fps:.2f} | Resolution: {width}x{height} | Frames: {total_frames}")
    print(f"Re-classify each track every {args.classify_every} frames "
          f"({args.classify_every / fps:.1f}s)\n")

    output = args.output or (os.path.splitext(os.path.basename(args.input))[0] + "_annotated.mp4")
    writer, output = open_writer(output, fps, (width, height))

    tracks = {}  # track_id -> TrackState
    frame_idx = 0
    n_dets = 0
    n_cls = 0
    cls_time = 0.0
    t_start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames > 0 and frame_idx >= args.max_frames):
                break

            result = detector.track(
                frame, persist=True, tracker="bytetrack.yaml",
                classes=args.classes, conf=args.det_conf, verbose=False,
            )[0]

            dets = []
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)

                for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    bw, bh = x2 - x1, y2 - y1
                    px, py = int(bw * args.pad), int(bh * args.pad)
                    cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
                    cx2, cy2 = min(width, x2 + px), min(height, y2 + py)

                    det = {"tid": tid, "box": (x1, y1, x2, y2), "crop": None, "label": None, "conf": 0.0}
                    if min(bw, bh) >= args.min_crop:
                        det["crop"] = frame[cy1:cy2, cx1:cx2]
                    dets.append(det)
                    n_dets += 1

                    if tid is not None:
                        st = tracks.setdefault(tid, TrackState(len(class_names)))
                        st.frames_seen += 1

            pending = []
            for det in dets:
                if det["crop"] is None:
                    continue
                tid = det["tid"]
                if tid is not None:
                    st = tracks[tid]
                    due = st.n_cls == 0 or (frame_idx - st.last_cls_frame) >= args.classify_every
                else:
                    due = frame_idx % args.classify_every == 0  # no ID: can't cache, so be sparse
                if due:
                    pending.append(det)

            if pending:
                t0 = time.time()
                probs = classifier.predict_batch([d["crop"] for d in pending])
                cls_time += time.time() - t0
                n_cls += len(pending)

                for det, p in zip(pending, probs):
                    if det["tid"] is not None:
                        tracks[det["tid"]].update(p, frame_idx)
                    else:
                        idx = int(p.argmax())
                        det["label"], det["conf"] = class_names[idx], float(p[idx])

            for det in dets:
                x1, y1, x2, y2 = det["box"]
                tid = det["tid"]
                color = track_color(tid)

                label, conf = det["label"], det["conf"]
                if tid is not None:
                    idx, conf = tracks[tid].best()
                    label = class_names[idx] if idx is not None else None

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                prefix = f"#{tid} " if tid is not None else ""
                if label is None:
                    text = f"{prefix}car"
                elif conf < args.min_conf:
                    text = f"{prefix}? {label} ({conf:.2f})"
                else:
                    text = f"{prefix}{label} ({conf:.2f})"
                draw_label(frame, x1, y1, text, color)

            writer.write(frame)
            frame_idx += 1

            if args.show:
                cv2.imshow("video_tester (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    break

            if frame_idx % 30 == 0:
                elapsed = time.time() - t_start
                speed = frame_idx / elapsed
                pct = f"{frame_idx / total_frames * 100:5.1f}%" if total_frames > 0 else "  n/a "
                eta = f"{(total_frames - frame_idx) / speed:.0f}s" if total_frames > 0 and speed > 0 else "?"
                print(f"Frame {frame_idx}/{total_frames} ({pct}) | cars in frame: {len(dets)} | "
                      f"classifications: {n_cls} | {speed:.1f} fps | ETA {eta}")

    except KeyboardInterrupt:
        print("\nInterrupted; saving what was processed so far...")
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print(f"Frames processed:      {frame_idx}")
    print(f"Car detections:        {n_dets}")
    print(f"Unique tracks:         {len(tracks)}")
    print(f"Classifications:       {n_cls}"
          + (f" (avg {cls_time / n_cls * 1000:.0f} ms each)" if n_cls else ""))
    print(f"Total time:            {elapsed:.1f}s")
    print(f"Output:                {os.path.abspath(output)}")
    print("=" * 50)

    print_summary(tracks, class_names, args.csv)


if __name__ == "__main__":
    try:
        run(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
