"""
Custom exception classes for fish tracking system.

Provides specific exception types for different failure modes, enabling
better error handling and debugging.

Author: Refactored from original code by Monica Coraggioso
"""


class TrackingError(Exception):
    """Base exception for all tracking-related errors."""

    pass


class FishLostError(TrackingError):
    """Raised when fish tracking is lost (no contours detected).

    This typically occurs when:
    - Fish moves out of frame
    - Poor image quality prevents detection
    - Threshold settings are incorrect
    """

    def __init__(self, frame=None, message="Fish tracking lost"):
        self.frame = frame
        if frame is not None:
            message = f"{message} at frame {frame}"
        super().__init__(message)


class SegmentTrackingError(TrackingError):
    """Raised when tail/head segment tracking fails.

    This can occur when:
    - ROI contains no valid pixels
    - Segment direction changes too sharply (exceeds scalar product threshold)
    - Tracking reaches image boundary
    """

    def __init__(self, segment=None, frame=None, message="Segment tracking failed"):
        self.segment = segment
        self.frame = frame
        details = []
        if segment is not None:
            details.append(f"segment {segment}")
        if frame is not None:
            details.append(f"frame {frame}")
        if details:
            message = f"{message} at {', '.join(details)}"
        super().__init__(message)


class RegistrationError(TrackingError):
    """Raised when image registration fails.

    This can occur when:
    - Phase cross-correlation fails to converge
    - Rotation angle cannot be determined
    - Registration error exceeds acceptable threshold
    """

    def __init__(self, frame=None, error_value=None, message="Image registration failed"):
        self.frame = frame
        self.error_value = error_value
        if frame is not None:
            message = f"{message} at frame {frame}"
        if error_value is not None:
            message = f"{message} (error: {error_value:.4f})"
        super().__init__(message)


class SleapDataError(TrackingError):
    """Raised when SLEAP pose data is invalid or missing.

    This can occur when:
    - Confidence scores below threshold
    - Missing keypoints
    - Malformed data files
    """

    def __init__(self, frame=None, keypoint=None, message="SLEAP data error"):
        self.frame = frame
        self.keypoint = keypoint
        details = []
        if frame is not None:
            details.append(f"frame {frame}")
        if keypoint is not None:
            details.append(f"keypoint '{keypoint}'")
        if details:
            message = f"{message} at {', '.join(details)}"
        super().__init__(message)


class ConfigurationError(Exception):
    """Raised when configuration is invalid.

    This can occur when:
    - Required parameters are missing
    - Parameter values are out of valid range
    - YAML file is malformed
    """

    pass


class PreprocessingError(TrackingError):
    """Raised when image preprocessing fails.

    This can occur when:
    - Invalid image format or dimensions
    - Gaussian filter parameters invalid
    - Threshold operation fails
    """

    def __init__(self, message="Image preprocessing failed", details=None):
        if details:
            message = f"{message}: {details}"
        super().__init__(message)
