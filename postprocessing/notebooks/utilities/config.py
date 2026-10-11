"""
Configuration management for fish tracking experiments.

This module provides a centralized configuration system to replace scattered
magic numbers throughout the codebase. Configuration values are organized by
functional area and can be loaded from/saved to YAML files.

Author: Refactored from original code by Monica Coraggioso
"""
from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml
from pathlib import Path


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing operations.

    Attributes
    ----------
    gaussian_sigma : float
        Standard deviation for Gaussian blur filter (noise reduction)
    downsampling : int
        Factor by which to downsample images for faster processing
    th_img : int
        Threshold value for binary image conversion (0-255)
    noise_removal_area_min : float
        Minimum contour area (pixels) to keep during noise removal
    noise_removal_area_max : float
        Maximum contour area (pixels) to keep during noise removal
    """
    gaussian_sigma: float = 1.0
    downsampling: int = 2
    th_img: int = 100
    noise_removal_area_min: float = 10.0
    noise_removal_area_max: float = 5000.0


@dataclass
class TrackingConfig:
    """Configuration for fish body tracking algorithms.

    Attributes
    ----------
    n_segments : int
        Number of segments to divide the fish body/tail into
    window_radius : float
        Radius (pixels) of search window for next segment detection
    segment_length : float
        Expected length of each segment (pixels). If None, computed from fish_length/n_segments
    scalar_product_threshold : float
        Minimum dot product (0-1) between consecutive segment directions.
        Higher values enforce straighter tracking paths.
    """
    n_segments: int = 20
    window_radius: float = 5.0
    segment_length: Optional[float] = None
    scalar_product_threshold: float = 0.5


@dataclass
class RegistrationConfig:
    """Configuration for head image registration.

    Attributes
    ----------
    dcrop : int
        Cropping distance (pixels) around head region
    error_threshold : float
        Maximum allowed registration error before applying rotation correction
    upsample_factor_initial : int
        Upsampling factor for initial phase cross-correlation
    upsample_factor_refined : int
        Upsampling factor for refined phase cross-correlation after rotation
    angle_search_range : float
        Range in degrees (+/-) to search for rotation angle
    angle_search_step : float
        Step size in degrees for coarse angle search
    """
    dcrop: int = 15
    error_threshold: float = 0.2
    upsample_factor_initial: int = 100
    upsample_factor_refined: int = 50
    angle_search_range: float = 95.0
    angle_search_step: float = 1.0


@dataclass
class SleapConfig:
    """Configuration for SLEAP pose estimation integration.

    Attributes
    ----------
    th_score : float
        Minimum confidence threshold (0-1) for accepting a pose detection
    th_score_points : float
        Minimum confidence threshold for individual keypoint detections
    alignment_tolerance : float
        Maximum distance (pixels) for considering points aligned
    """
    th_score: float = 0.7
    th_score_points: float = 0.5
    alignment_tolerance: float = 1.5


@dataclass
class ExperimentConfig:
    """Root configuration combining all subsystems.

    This is the main configuration class that should be instantiated for
    experiments. It combines all specialized configs and provides methods
    for loading from and saving to YAML files.

    Attributes
    ----------
    preprocessing : PreprocessingConfig
        Image preprocessing parameters
    tracking : TrackingConfig
        Body/tail tracking parameters
    registration : RegistrationConfig
        Head registration parameters
    sleap : SleapConfig
        SLEAP integration parameters
    experiment_name : str
        Optional name for this experiment configuration

    Examples
    --------
    # Create with defaults
    config = ExperimentConfig()

    # Create with custom values
    config = ExperimentConfig(
        tracking=TrackingConfig(n_segments=30, window_radius=7.0),
        experiment_name="high_resolution_tracking"
    )

    # Load from YAML
    config = ExperimentConfig.from_yaml("my_config.yaml")

    # Save to YAML
    config.to_yaml("my_config.yaml")

    # Access nested config
    print(config.tracking.n_segments)
    """
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    sleap: SleapConfig = field(default_factory=SleapConfig)
    experiment_name: str = "default"

    @classmethod
    def from_yaml(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from a YAML file.

        Parameters
        ----------
        filepath : str
            Path to YAML configuration file

        Returns
        -------
        ExperimentConfig
            Configuration object with values loaded from file

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist
        yaml.YAMLError
            If the file is not valid YAML
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        # Reconstruct nested dataclasses
        preprocessing = PreprocessingConfig(**data.get('preprocessing', {}))
        tracking = TrackingConfig(**data.get('tracking', {}))
        registration = RegistrationConfig(**data.get('registration', {}))
        sleap = SleapConfig(**data.get('sleap', {}))

        return cls(
            preprocessing=preprocessing,
            tracking=tracking,
            registration=registration,
            sleap=sleap,
            experiment_name=data.get('experiment_name', 'default')
        )

    def to_yaml(self, filepath: str, include_comments: bool = True) -> None:
        """Save configuration to a YAML file.

        Parameters
        ----------
        filepath : str
            Path where YAML configuration file will be saved
        include_comments : bool, default=True
            If True, includes descriptive comments in the output file

        Notes
        -----
        Existing files will be overwritten without warning.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dictionary
        config_dict = {
            'experiment_name': self.experiment_name,
            'preprocessing': asdict(self.preprocessing),
            'tracking': asdict(self.tracking),
            'registration': asdict(self.registration),
            'sleap': asdict(self.sleap)
        }

        with open(filepath, 'w') as f:
            if include_comments:
                f.write("# Fish Tracking Experiment Configuration\n")
                f.write(f"# Automatically generated configuration file\n\n")
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert configuration to a nested dictionary.

        Returns
        -------
        dict
            Dictionary representation of all configuration values
        """
        return {
            'experiment_name': self.experiment_name,
            'preprocessing': asdict(self.preprocessing),
            'tracking': asdict(self.tracking),
            'registration': asdict(self.registration),
            'sleap': asdict(self.sleap)
        }

    def validate(self) -> list[str]:
        """Validate configuration values and return any warnings.

        Returns
        -------
        list of str
            List of warning messages. Empty list if all values are valid.

        Examples
        --------
        config = ExperimentConfig()
        warnings = config.validate()
        if warnings:
            for warning in warnings:
                print(f"Warning: {warning}")
        """
        warnings = []

        # Validate preprocessing
        if self.preprocessing.gaussian_sigma < 0:
            warnings.append("gaussian_sigma should be non-negative")
        if self.preprocessing.downsampling < 1:
            warnings.append("downsampling should be >= 1")
        if not (0 <= self.preprocessing.th_img <= 255):
            warnings.append("th_img should be in range [0, 255]")

        # Validate tracking
        if self.tracking.n_segments < 1:
            warnings.append("n_segments should be >= 1")
        if self.tracking.window_radius < 1:
            warnings.append("window_radius should be >= 1")
        if not (0 <= self.tracking.scalar_product_threshold <= 1):
            warnings.append("scalar_product_threshold should be in range [0, 1]")

        # Validate registration
        if self.registration.dcrop < 0:
            warnings.append("dcrop should be non-negative")
        if self.registration.error_threshold < 0:
            warnings.append("error_threshold should be non-negative")

        # Validate SLEAP
        if not (0 <= self.sleap.th_score <= 1):
            warnings.append("th_score should be in range [0, 1]")
        if not (0 <= self.sleap.th_score_points <= 1):
            warnings.append("th_score_points should be in range [0, 1]")

        return warnings
