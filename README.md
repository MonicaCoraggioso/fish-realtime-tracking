# fish-realtime-tracking

**Closed-loop, real-time tracking system for freely swimming larval fish, with a custom YOLOv8 detector and a SLEAP-based posture pipeline.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17650136.svg)](https://doi.org/10.5281/zenodo.17650136)
[![Paper](https://img.shields.io/badge/Science%20Advances-10.1126%2Fsciadv.aec6922-1f6feb)](https://doi.org/10.1126/sciadv.aec6922)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
[![tests](https://github.com/MonicaCoraggioso/fish-realtime-tracking/actions/workflows/tests.yml/badge.svg)](https://github.com/MonicaCoraggioso/fish-realtime-tracking/actions/workflows/tests.yml)

I designed and built this software to run the behavioural experiments of
[*A sensorimotor instability drives a locomotor transition during fish development*](https://doi.org/10.1126/sciadv.aec6922).
It follows a single *Danionella* larva (a traslucent fish, few mm long) as it swims freely across a large arena. A camera running at 250 fps sits on a motorised CNC gantry that keeps the fish in view, and visual stimuli are presented in closed loop.

The analysis of the resulting data is in the companion repository
**[locomotor-transition-development](https://github.com/MonicaCoraggioso/locomotor-transition-development)**.

---

## Pipeline

```
 ┌────────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐
 │ acquisition/           │   │ yolo/                  │   │ postprocessing/        │   │ data_processing/       │
 │ 250 fps camera capture │──▶│ custom YOLOv8 detector │   │ SLEAP keypoints +      │──▶│ quality filters,       │
 │ real-time tracking     │◀──│ (128×128 crops, GPU)   │   │ spline fit + image     │   │ spline regularisation, │
 │ CNC gantry follows fish│   └────────────────────────┘   │ registration → lab     │   │ merged INFO/KINE/POSE  │
 │ projected stimuli      │──────── raw videos ──────────▶ │ frame trajectories     │   │ HDF5 for analysis      │
 └────────────────────────┘                                └────────────────────────┘   └────────────────────────┘
```

| Module | What it does | Key tech |
|---|---|---|
| [`acquisition/`](acquisition/) | Real-time loop: frame grab → background subtraction → detection → head direction → motor command → stimulus update. Multi-threaded capture, tracking and motor control. | OpenCV, PyTorch/Ultralytics, PySerial (G-code), PsychoPy, FLIR Spinnaker |
| [`yolo/`](yolo/) | Building the annotated dataset, training YOLOv8 for robust detection (fish near walls, dust, varied postures and sizes), inference demo. | Ultralytics YOLOv8, Roboflow |
| [`postprocessing/`](postprocessing/) | Offline posture estimation from recorded videos: SLEAP keypoints, midline spline, sub-pixel head registration, trajectory in the lab frame. | SLEAP (U-Net), scikit-image, Numba |
| [`data_processing/`](data_processing/) | Turns per-run outputs into the analysis-ready `INFO`, `KINE` and `POSE` tables used in the paper. | pandas, h5py, SciPy |
| [`hardware/`](hardware/) | Set-up schematic and Shapeoko (GRBL) notes. | |

## Highlights

- **Real time at 250 fps**: tracking, motor control and stimulus rendering run in parallel threads, so the fish stays centred under the camera over the whole arena.
- **Hybrid detection**: fast classical CV (thresholding, ellipse fit, tail segmentation) with a **custom-trained YOLOv8** fallback that recovers the fish when classical tracking fails (edges, dust, overlaps).
- **Learned posture**: a SLEAP single-instance U-Net trained on 702 hand-labelled frames gives body keypoints, which are refined with spline fitting and phase-correlation registration.
- **Reproducible configs**: post-processing parameters live in a documented YAML file (`postprocessing/config_default.yaml`).

## Quick start

Each module has its own environment because they run on different machines (Windows acquisition PC with GPU vs. analysis workstation).

```bash
git clone https://github.com/MonicaCoraggioso/fish-realtime-tracking.git
cd fish-realtime-tracking

# Train or evaluate the detector (no special hardware needed)
pip install -r yolo/requirements.txt
cd yolo && jupyter notebook train_yolov8.ipynb
```

See each folder's README for details. The acquisition code needs the actual hardware (FLIR camera + Spinnaker SDK, Shapeoko CNC, projector).

## Tests

Unit tests cover the hardware-independent parts: pose and kinematics cleaning (`data_processing/`), the post-processing configuration, and the calibration and watchdog helpers of the acquisition code. They run automatically on every push (GitHub Actions) and locally with:

```bash
pip install -r requirements-test.txt
pytest
```

## Data and model weights

Data are **not** stored in this repository. They are archived on Zenodo: [10.5281/zenodo.17650136](https://doi.org/10.5281/zenodo.17650136).

| Item | Where |
|---|---|
| YOLOv8 training set (122 annotated 128×128 frames) | included: `yolo/dataset/` |
| YOLOv8 weights `yolo8m_live_tracking.pt` (52 MB) | [GitHub Releases](https://github.com/MonicaCoraggioso/fish-realtime-tracking/releases) and Zenodo `Tracking_camera.zip` |
| SLEAP model weights `best_model.h5` | [GitHub Releases](https://github.com/MonicaCoraggioso/fish-realtime-tracking/releases) and Zenodo `Tracking_camera.zip` |
| Example raw recordings | Zenodo `Tracking_camera.zip` → `data/` |
| Processed datasets (`INFO`, `KINE`, `POSE`) | Zenodo `Data_processing.zip` |

## Citation

If you use this code, please cite the paper and the archive (see [`CITATION.cff`](CITATION.cff)):

> Coraggioso M., Demarchi L., Wong R., Dichio V., Chaumeton C., Panier T., Morvan-Dubois G., Goodhill G., Bormuth V., Debrégeas G. *A sensorimotor instability drives a locomotor transition during fish development.* *Science Advances* (2026). https://doi.org/10.1126/sciadv.aec6922

## Author and credits

**Monica Coraggioso** ([ORCID](https://orcid.org/0009-0004-2091-3592)), Laboratoire Jean Perrin, Sorbonne Université. Data quality check and dataset standardization by **Vito Dichio**.

## License

Code: [MIT](LICENSE). YOLO dataset: CC BY 4.0 (Roboflow export).
