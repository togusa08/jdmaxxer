# Car Video Testing Tools

Tools for testing our JDM car identification models on video. They detect cars
with YOLOv8, track them with ByteTrack, and identify the model.

Part of the AI Vehicle Identification project.

## Files

| File | Purpose |
|---|---|
| `video_tester.py` | Full pipeline: detect, track, classify, and write an annotated video |
| `tester.py` | YOLO-only check: draws boxes on cars, with no classification |


## Setup

    pip install -r requirements.txt

Download the trained checkpoints from
[Hugging Face](https://huggingface.co/togusa08/jdm-car-classifier) and place
them in the repo root as `best_b3.pth` and `best_b4.pth`.


## Important notes

- Checkpoints must be full training dicts containing `model_state_dict`.
- Preprocessing (224x224, ImageNet normalization) must match training.
- If the output video is empty or won't open on Windows, try the `XVID`
  codec with a `.avi` extension.
