"""
Collection of functions useful for reading SLEAP tracking data.

Author: Monica Coraggioso
"""

import numpy as np
import h5py

def h5tree_view(file):
    """Display the structure and attributes of an HDF5 file in a tree-like format."""
    import h5py
    assert isinstance(file, h5py.File)

    def view_h5attributes(obj, depth=0):
        atts = obj.attrs
        prefix = "│   " * depth
        for i, k in enumerate(atts.keys()):
            branch = "└──" if i == len(atts) - 1 else "├──"
            try:
                val = atts[k].decode("utf-8")
            except (UnicodeDecodeError, AttributeError):
                val = atts[k]
            print(f"{prefix}{branch} 🏷️{k} = `{val}`")

    def view_h5object(obj, depth=0):
        name = obj.name.split("/")[-1]
        prefix = "│   " * depth + "├──"
        if isinstance(obj, h5py.Group):
            print(f"{prefix}📁{name}")
            view_h5attributes(obj, depth + 1)
            for k in obj:
                view_h5object(obj[k], depth + 1)
        else:
            print(f"{prefix}🔢{name} ⚙️{obj.shape}{obj.dtype}")
            view_h5attributes(obj, depth + 1)

    print(".", file.filename)
    for k in file:
        view_h5object(file[k], depth=0)


def read_sleap_labels(filename, show_details=False):
    """Load tracked body part locations and instance scores from a SLEAP HDF5 file."""
    with h5py.File(filename, "r") as f:
        if show_details:
            h5tree_view(f)

        # Read datasets
        dset_names = list(f.keys())
        locations = f["tracks"][:].T  # shape: (nodes, 2, frames, instances) --> .T -> (frames, nodes, 2, instances)
        node_names = [n.decode() for n in f["node_names"][:]]
        instance_scores = f['instance_scores'][:]
        point_scores = f['point_scores'][:]
        
        if 'h' in node_names:
            print('New SLEAP version')
            new_version = True

        if show_details:
            print(f"\n===filename===\n{filename}")
            print("\n===HDF5 datasets===\n", dset_names)
            print("\n===locations data shape===\n", locations.shape)
            print("\n===nodes===")
            for i, name in enumerate(node_names):
                print(f"{i}: {name}")

    # Map node names to their index
    node_index = {name: idx for idx, name in enumerate(node_names)}
    
    # Define required nodes and fallback for optional ones
    #def get_node_data(name):
    #    if name in node_index:
    #        return locations[:, node_index[name], :, :]
    #    else:
    #        print(f"[WARN] Node '{name}' not found in file. Filling with NaNs.")
    #        return np.full((locations.shape[0], 2, locations.shape[3]), np.nan)
    
    def get_node_data(name):
        if name in node_index:
            if locations.ndim == 3:  # shape: (frames, nodes, 2)
                return locations[:, node_index[name], :]
            elif locations.ndim == 4:  # shape: (tracks, frames, nodes, 2)
                return locations[ :, node_index[name], :, :]  # assume track 0
        else:
            print(f"[WARN] Node '{name}' not found in file. Filling with NaNs.")
            return np.full((locations.shape[1], 2), np.nan)  # fallback shape: (frames, 2)
        
    if not new_version:
        # Load all body part data, safely for optional ones
        head_loc = get_node_data("head")
        t1_loc = get_node_data("t1")
        t2_loc = get_node_data("t2")
        t3_loc = get_node_data("t3")
        t4_loc = get_node_data("t4")
        fl1_loc = get_node_data("fl1")
        fl2_loc = get_node_data("fl2")
        fr1_loc = get_node_data("fr1")
        fr2_loc = get_node_data("fr2")
    else:
        head_loc = get_node_data("h")
        t1_loc = get_node_data("t1")
        t2_loc = get_node_data("t2")
        t3_loc = get_node_data("t3")
        t4_loc = get_node_data("t4")
        fl1_loc, fl2_loc, fr1_loc, fr2_loc = np.nan, np.nan, np.nan, np.nan

    print(f"\nlocations loaded with shape {locations.shape}")    
    print(f"instance_scores loaded with shape {instance_scores.shape}") 
    
    return instance_scores, point_scores, head_loc, t1_loc, t2_loc, t3_loc, t4_loc, fl1_loc, fl2_loc, fr1_loc, fr2_loc


def quantities_from_sleap(instance_scores, th_score, head_loc, t1_loc, t2_loc, t3_loc, t4_loc):
    """Compute various posture quantities from SLEAP predictions."""
    # Select instances above score threshold
    id_score = np.where((instance_scores[0] > th_score) & ~np.isnan(instance_scores[0]))

    # Distance head–t1
    head_L = np.linalg.norm(head_loc - t1_loc, axis=1)
    mean_head_L = round(np.nanmean(head_L[id_score]), 2)
    print(f'mean_head_L: {mean_head_L}')

    # Full tail profile length: sum of t1–t4 segment lengths
    profile_L = (
        head_L +
        np.linalg.norm(t2_loc - t1_loc, axis=1) +
        np.linalg.norm(t3_loc - t2_loc, axis=1) +
        np.linalg.norm(t4_loc - t3_loc, axis=1)
    )
    mean_fish_L = round(np.nanmean(profile_L[id_score]), 2)
    print(f'mean_fish_L: {mean_fish_L}')

    # Head and tail directions
    head_direction = np.arctan2(head_loc[:, 1, :] - t1_loc[:, 1, :],
                                head_loc[:, 0, :] - t1_loc[:, 0, :])
    tail_direction = np.arctan2(t1_loc[:, 1, :] - t2_loc[:, 1, :],
                                t1_loc[:, 0, :] - t2_loc[:, 0, :])

    # Angle unwrapping (to avoid jumps across 360°)
    head_direction_unwrap = np.full_like(head_direction, np.nan)
    for i in range(head_direction.shape[1]):
        valid = ~np.isnan(head_direction[:, i])
        head_direction_unwrap[valid, i] = np.rad2deg(np.unwrap(head_direction[valid, i]))

    return id_score, head_L, mean_head_L, profile_L, mean_fish_L, head_direction, tail_direction, head_direction_unwrap