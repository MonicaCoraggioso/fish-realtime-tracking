"""
Core tracking functions for fish tail and head segmentation.

This module provides the fundamental tracking algorithms used across multiple
tracking modules. These functions implement segment-based tracking inspired by
Stytra (https://www.nature.com/articles/s41592-019-0399-2).

The tracking works by iteratively finding the center of mass within a region of
interest (ROI) around the expected position of the next segment, either along
the tail or head. Two ROI types are provided:
- Circular ROI: for tail tracking (next_segment)
- Rectangular ROI: for head tracking (next_segment_head)

All core segment-finding functions use Numba JIT compilation for performance.

Author: Monica Coraggioso
"""
import numpy as np
from numba import njit


@njit
def next_segment(im, x0, y0, dx, dy, r, segment_length, scalar_product_threshold=0.5):
    """Find the next segment point along the fish tail using circular ROI.

    This function segments the tail by finding the center of mass within a circular
    region of interest around the predicted next point. The algorithm checks that
    the direction remains consistent using a scalar product threshold.

    Inspired by Stytra tail tracking (Štih et al., 2019).

    Parameters
    ----------
    im : 2D np.array, dtype=float32
        Binary or grayscale image with fish visible
    x0, y0 : float
        Current position (e.g., centroid or previous segment point)
    dx, dy : float
        Direction vector of current segment (pointing to expected next position)
    r : float
        Radius of the circular ROI for searching the next point
    segment_length : float
        Length of each segment (typically fish_length / n_segments)
    scalar_product_threshold : float, default=0.5
        Minimum normalized dot product between consecutive segments to accept
        the new direction. Range [0, 1]. Higher values enforce straighter paths.

    Returns
    -------
    x_new, y_new : float
        Position of the next segment point, or np.nan if tracking fails
    dx_new, dy_new : float
        Direction vector of the new segment, or np.nan if tracking fails

    Notes
    -----
    Returns np.nan values if:
    - ROI is empty or outside image bounds
    - No intensity found in ROI (acc == 0)
    - Direction changes too sharply (fails scalar product test)
    """
    r2 = r**2
    y_max, x_max = im.shape
    xs = min(max(int(round(x0 + dx - r)), 0), x_max)
    xl = min(max(int(round(x0 + dx + r)), 0), x_max)
    ys = min(max(int(round(y0 + dy - r)), 0), y_max)
    yl = min(max(int(round(y0 + dy + r)), 0), y_max)

    if xs == xl or ys == yl:
        return np.nan, np.nan, np.nan, np.nan

    acc = 0.0
    acc_x = 0.0
    acc_y = 0.0
    for x in range(xs, xl):
        for y in range(ys, yl):
            lx = (x0 + dx - x)**2
            ly = (y0 + dy - y)**2
            if lx + ly <= r2:
                acc_x += x * im[y, x]
                acc_y += y * im[y, x]
                acc += im[y, x]

    if acc == 0:
        return np.nan, np.nan, np.nan, np.nan

    dx_new = acc_x / acc - x0
    dy_new = acc_y / acc - y0
    norm = np.sqrt(dx_new**2 + dy_new**2)

    if norm == 0:
        return np.nan, np.nan, np.nan, np.nan

    dx_new = dx_new / norm * segment_length
    dy_new = dy_new / norm * segment_length

    if (dx * dx_new + dy * dy_new) / segment_length**2 < scalar_product_threshold:
        return np.nan, np.nan, np.nan, np.nan

    return x0 + dx_new, y0 + dy_new, dx_new, dy_new


@njit
def next_segment_head(im, x0, y0, dx, dy, r, segment_length, scalar_product_threshold=0.5):
    """Find the next segment point along the fish head using rectangular ROI.

    Similar to next_segment but uses a rectangular ROI oriented perpendicular to
    the direction of travel. This is more appropriate for head tracking where the
    fish body is wider.

    Parameters
    ----------
    im : 2D np.array, dtype=float32
        Binary or grayscale image with fish visible
    x0, y0 : float
        Current position
    dx, dy : float
        Direction vector of current segment
    r : float
        Width parameter for the rectangular ROI (half-width perpendicular to direction)
    segment_length : float
        Length of each segment
    scalar_product_threshold : float, default=0.5
        Minimum normalized dot product between consecutive segments

    Returns
    -------
    x_new, y_new : float
        Position of the next segment point, or np.nan if tracking fails
    dx_new, dy_new : float
        Direction vector of the new segment, or np.nan if tracking fails

    Notes
    -----
    The rectangular ROI is defined by:
    - Length: r/2 along the direction of motion
    - Width: 2*r perpendicular to the direction of motion
    Additionally, a circular constraint is still applied within the rectangle.
    """
    # Calculate perpendicular vector to (dx, dy)
    perp_dx = -dy
    perp_dy = dx

    # Normalize perpendicular vector to length r
    perp_norm = np.sqrt(perp_dx**2 + perp_dy**2)
    perp_dx = perp_dx / perp_norm * r
    perp_dy = perp_dy / perp_norm * r

    # Normalize the direction vector to length r/2
    direction_norm = np.sqrt(dx**2 + dy**2)
    half_dx = dx / direction_norm * (r / 2)
    half_dy = dy / direction_norm * (r / 2)

    # Calculate rectangle corners
    x_center = x0 + dx
    y_center = y0 + dy
    corners = [
        (x_center + half_dx + perp_dx, y_center + half_dy + perp_dy),
        (x_center + half_dx - perp_dx, y_center + half_dy - perp_dy),
        (x_center - half_dx + perp_dx, y_center - half_dy + perp_dy),
        (x_center - half_dx - perp_dx, y_center - half_dy - perp_dy)
    ]

    # Find min and max x and y from corners
    xs = min([int(round(corner[0])) for corner in corners])
    xl = max([int(round(corner[0])) for corner in corners])
    ys = min([int(round(corner[1])) for corner in corners])
    yl = max([int(round(corner[1])) for corner in corners])

    # Clamping to image boundaries
    y_max, x_max = im.shape
    xs = max(min(xs, x_max - 1), 0)
    xl = max(min(xl, x_max - 1), 0)
    ys = max(min(ys, y_max - 1), 0)
    yl = max(min(yl, y_max - 1), 0)

    if xs == xl or ys == yl:
        return np.nan, np.nan, np.nan, np.nan

    acc = 0.0
    acc_x = 0.0
    acc_y = 0.0
    for x in range(xs, xl + 1):
        for y in range(ys, yl + 1):
            lx = (x0 + dx - x)**2
            ly = (y0 + dy - y)**2
            if lx + ly <= r**2:  # Circular constraint within rectangle
                acc_x += x * im[y, x]
                acc_y += y * im[y, x]
                acc += im[y, x]

    if acc == 0:
        return np.nan, np.nan, np.nan, np.nan

    dx_new = acc_x / acc - x0
    dy_new = acc_y / acc - y0
    norm = np.sqrt(dx_new**2 + dy_new**2)

    if norm == 0:
        return np.nan, np.nan, np.nan, np.nan

    dx_new = dx_new / norm * segment_length
    dy_new = dy_new / norm * segment_length

    return x0 + dx_new, y0 + dy_new, dx_new, dy_new


def tracking_tail(x, y, orientation, img_bw, n_segments, window_radius, tail_length, segment_length):
    """Track tail points from fish centroid using iterative segment finding.

    Starting from the fish centroid, this function iteratively finds segment
    points along the tail by calling next_segment repeatedly.

    Parameters
    ----------
    x, y : float
        Fish position (centroid)
    orientation : float
        Tail orientation in radians (direction pointing from head to tail)
    img_bw : 2D np.array
        Binary or grayscale image with fish visible
    n_segments : int
        Number of segments to track along the tail
    window_radius : float
        Radius of circular ROI for searching each segment
    tail_length : float
        Total length of the tail in pixels
    segment_length : float
        Length of each segment (typically tail_length / n_segments)

    Returns
    -------
    tail_points : np.array, shape (n_segments+1, 2)
        Array of (x, y) coordinates for each segment point.
        First point is the starting position (x, y).
        Contains np.nan if fish is not detected or tracking fails.

    Notes
    -----
    Tracking stops early if next_segment returns np.nan, leaving remaining
    points as np.nan.
    """
    tail_points = np.full((n_segments + 1, 2), np.nan)

    # Only track if fish is detected
    if ~np.isnan(x) and ~np.isnan(orientation):
        initial_x, initial_y = x, y
        theta_tail = orientation

        initial_dx = segment_length * np.cos(theta_tail)
        initial_dy = segment_length * np.sin(theta_tail)

        tail_points[0] = [initial_x, initial_y]

        for s in range(1, n_segments + 1):
            initial_x, initial_y, initial_dx, initial_dy = next_segment(
                img_bw, initial_x, initial_y, initial_dx, initial_dy,
                window_radius, segment_length
            )

            if np.isnan(initial_x):
                break

            tail_points[s] = [initial_x, initial_y]

    return tail_points


def tracking_tail_extrapolate(x, y, orientation, img_bw, n_segments, window_radius, tail_length, segment_length):
    """Track tail points with linear extrapolation when tracking fails.

    Similar to tracking_tail, but when next_segment fails (returns np.nan),
    remaining points are linearly extrapolated based on the previous segments.
    This is useful for completing the tail tip when the image quality degrades.

    Parameters
    ----------
    x, y : float
        Fish position (centroid)
    orientation : float
        Tail orientation in radians
    img_bw : 2D np.array
        Binary or grayscale image with fish visible
    n_segments : int
        Number of segments to track along the tail
    window_radius : float
        Radius of circular ROI for searching each segment
    tail_length : float
        Total length of the tail in pixels
    segment_length : float
        Length of each segment

    Returns
    -------
    tail_points : np.array, shape (n_segments+1, 2)
        Array of (x, y) coordinates for each segment point.
        Points after tracking failure are linearly extrapolated.
        Contains np.nan if fish is not detected.

    Notes
    -----
    Extrapolation uses the vector between the last two successfully tracked
    points to estimate remaining points linearly.
    """
    tail_points = np.full((n_segments + 1, 2), np.nan)

    # Only track if fish is detected
    if ~np.isnan(x) and ~np.isnan(orientation):
        initial_x, initial_y = x, y
        theta_tail = orientation

        initial_dx = segment_length * np.cos(theta_tail)
        initial_dy = segment_length * np.sin(theta_tail)

        tail_points[0] = [initial_x, initial_y]

        for s in range(1, n_segments + 1):
            initial_x, initial_y, initial_dx, initial_dy = next_segment(
                img_bw, initial_x, initial_y, initial_dx, initial_dy,
                window_radius, segment_length
            )

            if np.isnan(initial_x):
                # Linear extrapolation for remaining points
                tail_points[s:] = np.nan
                for sp in range(s, n_segments + 1):
                    if sp == 1:
                        tail_points[1] = (initial_x + initial_dx, initial_y + initial_dy)
                    else:
                        tail_points[sp] = tail_points[sp - 1] + tail_points[sp - 1] - tail_points[sp - 2]
                break

            tail_points[s] = [initial_x, initial_y]

    return tail_points


def tracking_head_extrapolate(x, y, orientation, img_bw, n_segments, window_radius, head_length, segment_length):
    """Track head points with linear extrapolation when tracking fails.

    Same as tracking_tail_extrapolate but uses next_segment_head with rectangular
    ROI, which is more appropriate for head tracking.

    Parameters
    ----------
    x, y : float
        Fish position (centroid or head position)
    orientation : float
        Head orientation in radians (direction pointing from centroid to head)
    img_bw : 2D np.array
        Binary or grayscale image with fish visible
    n_segments : int
        Number of segments to track along the head
    window_radius : float
        Width parameter for rectangular ROI
    head_length : float
        Total length to track along the head in pixels
    segment_length : float
        Length of each segment

    Returns
    -------
    head_points : np.array, shape (n_segments+1, 2)
        Array of (x, y) coordinates for each segment point.
        Points after tracking failure are linearly extrapolated.
        Contains np.nan if fish is not detected.
    """
    head_points = np.full((n_segments + 1, 2), np.nan)

    # Only track if fish is detected
    if ~np.isnan(x) and ~np.isnan(orientation):
        initial_x, initial_y = x, y
        initial_dx = segment_length * np.cos(orientation)
        initial_dy = segment_length * np.sin(orientation)

        head_points[0] = [initial_x, initial_y]

        for s in range(1, n_segments + 1):
            initial_x, initial_y, initial_dx, initial_dy = next_segment_head(
                img_bw, initial_x, initial_y, initial_dx, initial_dy,
                window_radius, segment_length
            )

            if np.isnan(initial_x):
                # Linear extrapolation for remaining points
                head_points[s:] = np.nan
                for sp in range(s, n_segments + 1):
                    if sp == 1:
                        head_points[1] = (initial_x + initial_dx, initial_y + initial_dy)
                    else:
                        head_points[sp] = head_points[sp - 1] + head_points[sp - 1] - head_points[sp - 2]
                break

            head_points[s] = [initial_x, initial_y]

    return head_points
