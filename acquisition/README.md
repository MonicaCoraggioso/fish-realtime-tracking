# Acquisition: real-time tracking and closed-loop control

A comprehensive real-time fish tracking and closed-loop control system for behavioral experiments. This system integrates high-speed video capture, computer vision tracking, motorized platform control, and visual stimulus presentation to study freely swimming fish behavior.

## Overview

The system performs real-time tracking of fish position and orientation at high frame rates (250 fps), and uses this information to:
- Control a motorized platform (Shapeoko CNC) in closed-loop
- Display visual stimuli via projector or second screen
- Log experimental data for post-processing analysis

The tracking combines traditional computer vision techniques (centroid, ellipse fitting) with deep learning-based object detection (YOLO).

## System Architecture

### Hardware Components
- **Point Grey Camera**: High-speed video capture (250 fps, 608x600 resolution)
- **Shapeoko CNC Machine**: Motorized platform for closed-loop position control
- **Projector/Second Screen**: Visual stimulus presentation using PsychoPy
- **Computer**: Python/Jupyter environment with CUDA support for GPU acceleration

### Software Components

#### Main Notebook
- **`freely_swimming.ipynb`**: Complete experimental workflow with comprehensive comments

#### Utilities Package (`utilities/`)
- **`camera_PointGrey.py`**: Camera initialization, capture, and cleanup
- **`tracking.py`**: Traditional CV tracking (centroid, ellipse fitting, tail segmentation)
- **`tracking_yolo.py`**: YOLO-based object detection and image cropping
- **`tracking_head_dir.py`**: Head direction detection and validation
- **`preprocessing.py`**: Image preprocessing (downsampling, Gaussian filtering, background subtraction, binarization)
- **`Shapeoko.py`**: CNC machine control via G-code (movement, homing, following)
- **`psychopy_projection.py`**: Visual stimulus generation and projection
- **`projector_and_Shapeoko.py`**: Manual control modes for projector and motor
- **`saving.py`**: Data saving utilities (NumPy arrays, HDF5 files, images)
- **`reading.py`**: Data loading utilities (HDF5, MP4, sorted file access)
- **`live_kinematic_parameters.py`**: Real-time kinematic parameter calculation
- **`mask.py`**: Region of interest (ROI) mask creation and management
- **`cleaning_edge.py`**: Edge cleaning and border artifact removal
- **`my_fit.py`**: Line fitting utilities for calibration
- **`sanity_test.py`**: System health checks, distance calculations, optional SMS notifications (credentials via environment variables)
- **`monitor_and_Shapeoko.py`**: Manual control and monitoring modes
- **`plotting.py`**: Visualization utilities for trajectories and data

## Installation

### Prerequisites

1. **Python 3.8+** (tested with Python 3.8.13)

2. **Point Grey/FLIR Spinnaker SDK**
   - Download from: https://www.flir.com/products/spinnaker-sdk/
   - Install the SDK appropriate for your operating system
   - The `rotpy` package requires the SDK to be installed first

3. **CUDA Toolkit** (for GPU acceleration with YOLO)
   - Required for YOLO model inference
   - Install CUDA 11.8+ and compatible cuDNN

### Python Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

**Note on rotpy**: After installing the Spinnaker SDK, install rotpy:
```bash
pip install rotpy
```

### Model Files

- **`yolo8m_live_tracking.pt`**: trained YOLOv8 weights for fish detection. Download them from the repository's GitHub Releases (or Zenodo) and place the file in this folder. Training code is in [`../yolo/`](../yolo/).

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `DATA_ROOT` | Root folder where experiment data are written | `D:\data_freely_swimming` |
| `SHAPEOKO_PORT` | Serial port of the Shapeoko controller | `COM5` |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, `ALERT_PHONE_NUMBER` | Optional SMS alert when the fish is lost / motor stops | unset → alerts disabled |

## Usage

### Basic Workflow

1. **Open the notebook**:
   ```bash
   jupyter notebook freely_swimming.ipynb
   ```

2. **Run setup cells**:
   - Configure experiment parameters (fish age, number, etc.)
   - Initialize camera, projector, and Shapeoko motor
   - Perform spatial calibration (pixels to mm conversion)
   - Load YOLO model

3. **Background setup**:
   - Capture and process background images
   - Create masks for region of interest
   - Set up visual stimulus patterns

4. **Run experiment**:
   - Execute the main experimental loop
   - System performs real-time tracking and control
   - Data is automatically saved

5. **Post-processing**:
   - Visualize trajectories and performance metrics
   - Analyze saved data

### Configuration

Key parameters to configure:
- **Camera settings**: Resolution, frame rate, offset (in notebook)
- **Tracking parameters**: Gaussian sigma, downsampling factor, threshold
- **Motor calibration**: Center position, speed limits
- **Projector settings**: Screen ID, resolution, physical dimensions
- **Data paths**: Experiment directory structure

## Data Organization

Data is automatically organized in the following structure:
```
data_freely_swimming/
└── fs_{age}dpf_{date}/
    └── fish_{number}/
        ├── positions.h5
        ├── timestamps.h5
        ├── motor_commands.h5
        └── video.mp4 (optional)
```

Where:
- `age`: Fish age in days post fertilization (dpf)
- `date`: Experiment date (DDMonYY format)
- `number`: Fish identification number

## Code Documentation

All code is fully documented with:
- **Module-level docstrings**: Explain module purpose and usage
- **Function docstrings**: NumPy-style documentation with parameters, returns, and examples
- **Type hints**: Function signatures include type annotations
- **Inline comments**: Explain complex logic and algorithms

## Key Features

- **High-speed tracking**: 250 fps video capture and processing
- **Dual tracking methods**: Traditional CV + YOLO deep learning
- **Closed-loop control**: Real-time motor adjustment based on fish position
- **Visual stimuli**: Programmable background patterns and projections
- **Comprehensive logging**: All data saved in HDF5 format for analysis
- **Robust error handling**: Fish loss detection and recovery
- **Multi-threaded architecture**: Parallel camera capture, tracking, and motor control

## System Requirements

### Minimum
- CPU: Multi-core processor (4+ cores recommended)
- RAM: 8 GB (16 GB recommended)
- GPU: NVIDIA GPU with CUDA support (for YOLO)
- Storage: SSD recommended for high-speed data writing

### Operating System
- Windows 10/11 (primary development platform)
- Linux (should work with appropriate drivers)
- macOS (may require additional setup for hardware)

## Troubleshooting

### Camera Issues
- Ensure Spinnaker SDK is properly installed
- Check camera connection and power
- Verify camera is not being used by another application

### Motor Control Issues
- Verify serial port connection (default: COM5 on Windows)
- Check Shapeoko is powered and homed
- Ensure G-code settings match your machine configuration

### YOLO/GPU Issues
- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU memory availability
- Model will fall back to CPU if GPU unavailable (slower)

A diagram of the code architecture is in [`docs/code_architecture.pdf`](docs/code_architecture.pdf).

## Citation

See the [main README](../README.md#citation).

## Author

**Monica Coraggioso**

## Acknowledgments

- YOLO implementation: Ultralytics (https://ultralytics.com)
- PsychoPy: Peirce et al. (2019) Behavior Research Methods
- Shapeoko: Carbide 3D (https://carbide3d.com)

## References

- Ellipse fitting algorithm inspired by: http://www.raphael.candelier.fr/
- Tail segmentation inspired by Stytra framework
