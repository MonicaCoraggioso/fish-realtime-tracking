# YOLOv8 fish detector

A custom **YOLOv8** model that finds a freely swimming *Danionella* larva in each frame. The real-time tracker in [`../acquisition/`](../acquisition/) uses it as a robust detector when classical thresholding fails: fish touching the arena wall, dust or imaging noise, unusual body postures, and the large change in body size during development.

Pre-trained models on Roboflow Universe did not handle these conditions, so I built a dedicated dataset and fine-tuned YOLOv8m on it.

## Contents

| File | Purpose |
|---|---|
| `create_dataset.ipynb` | Extracts frames from recorded experiments, preprocesses them (binarise, crop to 128×128) and exports them for labelling. Uses `../acquisition/utilities`. |
| `train_yolov8.ipynb` | Trains YOLOv8m on `dataset/`, then runs inference on single frames and full sequences, with an interactive viewer. |
| `dataset/` | YOLO-format dataset (`data.yaml`, `train/`, `valid/`, `test/`). |
| `docs/yolo_training_overview.pdf` | Slides summarising the training approach. |

## Dataset

- **122 annotated frames** (one class: `fish`), 128×128 px, exported from Roboflow ([`dc_freely` v4](https://universe.roboflow.com/ljpdanionella-behaviour/dc_freely/dataset/4)).
- Split: **106 train / 11 validation / 5 test**.
- Augmentation (3 versions per source frame in the training set): horizontal flip, 90° rotations.
- The frames were chosen to cover the hard cases: fish near the edges, a wide range of postures, dust, fish off-centre in the region of interest, and different developmental stages.

## Usage

```bash
pip install -r requirements.txt
jupyter notebook train_yolov8.ipynb
```

`train_yolov8.ipynb` uses the bundled `dataset/data.yaml` by default (paths in it are relative). To pull the dataset from Roboflow instead, set `ROBOFLOW_API_KEY` in your environment. The key is never written in the notebook.

Inference expects frames downscaled by 2 to the 128×128 training resolution, with a confidence threshold of 0.4. On the acquisition PC's GPU this takes about 11–13 ms per frame.

## Weights

The trained weights `yolo8m_live_tracking.pt` (52 MB) are attached to this repository's **GitHub Release** and archived on [Zenodo](https://doi.org/10.5281/zenodo.17650136). Put the file in `yolo/` (inference demo) or `acquisition/` (live tracking).

## License

The dataset is CC BY 4.0 (see `dataset/README.roboflow.txt`). The code is MIT.

## References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow: How to train YOLOv8 on a custom dataset](https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset)
