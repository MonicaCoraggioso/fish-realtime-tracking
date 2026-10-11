# YOLOv8 fish detector

A custom **YOLOv8m** model that finds a freely swimming *Danionella* larva in each frame. The real-time tracker in [`../acquisition/`](../acquisition/) combines it with fast classical image analysis: thresholding takes about 2 ms but fails with dust, reflections or a fish swimming along the arena wall, while YOLO stays reliable in those cases.

Pre-trained models on Roboflow Universe did not handle these conditions, so I built a dedicated dataset and trained a custom model. Full details are in Chapter 2.3 of my PhD thesis ([Coraggioso, 2025](https://theses.hal.science/tel-05569582)).

## Contents

| File | Purpose |
|---|---|
| `create_dataset.ipynb` | Extracts frames from recorded experiments, preprocesses them (binarise, crop to 128×128) and exports them for labelling. Uses `../acquisition/utilities`. |
| `train_yolov8.ipynb` | Trains YOLOv8m, then runs inference on single frames and full sequences, with an interactive viewer. |
| `evaluate.py` | Evaluates the weights on a labelled split: precision, recall, mAP, detection rate, centre error. |
| `dataset/` | YOLO-format dataset (`data.yaml`, `train/`, `valid/`, `test/`), see below. |
| `docs/yolo_training_overview.pdf` | Slides summarising the training approach. |

## Training data and settings

The final training procedure (thesis, Ch. 2.3) uses **797 manually labelled frames** (702 train / 63 validation / 32 test), curated to cover the hard cases met in experiments: fish near the tank edges, varied postures, dust and imaging noise, fish off-centre in the region of interest, and fish at different developmental stages.

- **Preprocessing:** frames binarised and cropped to **128×128 px**, which is key to fast inference.
- **Augmentation:** horizontal and vertical flips, 90° rotations, small rotations (±2°), Gaussian blur (up to 1.4 px), random noise (up to 0.1% of pixels).
- **Training:** YOLOv8m (Ultralytics, starting from the `yolov8m.pt` checkpoint), **100 epochs**, `imgsz=128`.

The **bundled `dataset/`** is a smaller export of the same kind of data (Roboflow [`dc_freely` v4](https://universe.roboflow.com/ljpdanionella-behaviour/dc_freely/dataset/4): 122 frames, 106 / 11 / 5), so you can run the training and the tests without downloading anything.

## How detections are used in real time

The rules below are implemented in [`acquisition/utilities/yolo_postprocessing.py`](../acquisition/utilities/yolo_postprocessing.py) and unit-tested in [`tests/`](../tests/):

1. Each camera frame is downscaled by 2 to the 128×128 YOLO input and run on the GPU; boxes are scaled back to camera pixels.
2. **Several detections** (dust particles, or a partly occluded fish split in two): keep the one closest to the fish position at the previous frame.
3. **No valid detection:** keep the previous position, with confidence 0.
4. **Near the arena edges**, where YOLO can be less confident, the YOLO position is fused with a kinematic prediction (previous position plus one step along the heading) using a confidence weight `w = 1 − exp(−conf / 0.1)`.
5. The fish is flagged as **lost** when the confidence drops below 0.03, which triggers the motor's search mode.

## Evaluation

**Online tracking performance** (thesis, Fig. 2.6). Run duration is the uninterrupted time a fish is tracked before detection is lost:

| Age (dpf) | Runs (N) | Mean run duration |
|---|---|---|
| 7–14 | 10 | 13 min |
| 14–21 | 22 | 20 min |
| 21–28 | 19 | 15 min |
| 28+ | 31 | 9 min |

On average the system tracks a fish continuously for about **15 minutes**, often more than 30 minutes for 14–28 dpf larvae. Older fish (28+ dpf) swim faster, which makes continuous detection harder and runs shorter.

**Inference speed:** about 11–13 ms per frame for YOLOv8m at 128×128 on the acquisition PC's GPU, compared with about 2 ms for the classical method. The 250 fps camera and the ≥50 Hz tracking rate needed to follow the fish are why both methods are combined.

**Before deployment**, the model was validated on previously recorded videos. Typical failure modes (multiple boxes caused by dust, split boxes caused by occlusion) are handled by the nearest-neighbour rule above.

**Offline metrics on a labelled split:**

```bash
pip install ultralytics
python evaluate.py --split test
```

**Automated checks** (`pytest`, run from the repository root):
- `tests/test_yolo_dataset.py`: one fish class, every image labelled, valid YOLO boxes, 128×128 images, and no augmented copy of a validation or test frame in the training set (no data leakage).
- `tests/test_yolo_postprocessing.py`: the real-time decision rules above.
- `tests/test_yolo_model.py`: the trained weights detect the fish on every test image (confidence ≥ 0.25) and their boxes overlap the ground truth (mean IoU > 0.5). These run locally once the weights are downloaded, and are skipped on GitHub Actions.

## Usage

```bash
pip install -r requirements.txt
jupyter notebook train_yolov8.ipynb
```

`train_yolov8.ipynb` uses the bundled `dataset/data.yaml` by default. To pull a dataset from Roboflow instead, set `ROBOFLOW_API_KEY` in your environment; the key is never written in the notebook.

## Weights

The trained weights `yolo8m_live_tracking.pt` (52 MB) are attached to this repository's **[GitHub Release](https://github.com/MonicaCoraggioso/fish-realtime-tracking/releases)** and archived on [Zenodo](https://doi.org/10.5281/zenodo.17650136). Put the file in `yolo/` (evaluation and tests) or `acquisition/` (live tracking).

## License

The dataset is CC BY 4.0 (see `dataset/README.roboflow.txt`). The code is MIT.

## References

- M. Coraggioso, *Multiscale locomotor dynamics in Danionella cerebrum*, PhD thesis, Sorbonne Université (2025), Chapter 2.3. https://theses.hal.science/tel-05569582
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow: How to train YOLOv8 on a custom dataset](https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset)
