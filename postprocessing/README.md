# Post-processing: posture and trajectory reconstruction

An offline pipeline that turns recorded experiment videos into analysis-ready posture and trajectory data. It is **self-contained**: it does not import anything from `acquisition/` or `yolo/`.

```
raw run (.h5 / .npy movie)
   │  convert_npy_to_mp4.ipynb           → MP4 movies for SLEAP labelling and inference
   ▼
SLEAP single-instance model             → body keypoints per frame
   │  sleap_registration_<condition>.ipynb
   ▼
spline fit of the midline + head image registration (phase correlation, rotation search)
   ▼
position (X, Y), heading φ and body shape in the laboratory frame → run{N}_postprocessing.h5
```

The output feeds [`../data_processing/`](../data_processing/).

## Contents

| Path | Description |
|---|---|
| `notebooks/convert_npy_to_mp4.ipynb` | Converts raw NumPy movies to MP4 and sets up the per-fish folder layout. |
| `notebooks/sleap_registration_freely_swimming.ipynb` | SLEAP keypoints + spline + registration for the main developmental dataset. |
| `notebooks/sleap_registration_{dark,lateral_line,viscosity,viscosity_zebrafish}.ipynb` | Same pipeline for the control experiments: darkness, lateral-line ablation, increased viscosity, zebrafish. |
| `notebooks/utilities/` | Reusable modules: tracking (`tracking_spline`, `tracking_registration`, `core_tracking`), preprocessing, I/O, plotting, config, logging. |
| `config_default.yaml` | All tunable parameters (preprocessing, tracking, registration, SLEAP thresholds), each documented. |
| `sleap_model/` | Training configuration and log of the SLEAP model (U-Net backbone, 702 labelled training frames, input 250×250 scaled ×0.75). |

## Environment

```bash
# With SLEAP (recommended)
conda env create -f environment-sleap.yml
conda activate sleap-pro

# Without SLEAP (spline tracking and registration only)
pip install -r requirements.txt
```

`environment-sleap.yml` targets Apple Silicon (`tensorflow-macos`). On Linux or Windows, replace those two lines with a standard `tensorflow` install, or follow the [SLEAP installation guide](https://sleap.ai/installation.html).

## Running

Start Jupyter from `notebooks/` so that `utilities` can be imported:

```bash
cd postprocessing/notebooks
jupyter notebook
```

Set the folder with raw recordings through the `RAW_DATA_DIR` environment variable, or edit `input_path` in the first cells. Load parameters with:

```python
from utilities.config import ExperimentConfig
config = ExperimentConfig.from_yaml('../config_default.yaml')
```

## Model weights

The trained SLEAP weights (`best_model.h5`, 24 MB) are attached to the GitHub Release and archived on [Zenodo](https://doi.org/10.5281/zenodo.17650136) (`Tracking_camera.zip` → `postprocessing/SLEAPforDc/`). Put them in `sleap_model/` next to `training_config.json`.

## Author

**Monica Coraggioso**
