"""
    Some useful functions for plotting
    
    Author: Monica Coraggioso
"""
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random

def plot_test_spline(ax, frame,myVideo, spline_avg, t1_loc,head_loc,t2_loc,t3_loc,t4_loc,discarded_frame_set,instance_scores):
    ax.imshow(myVideo[:,:,frame], origin='lower', cmap='gray')
    ax.plot(t1_loc[frame][0], t1_loc[frame][1], '.')
    ax.plot(head_loc[frame][0], head_loc[frame][1], '.')
    ax.plot(t2_loc[frame][0], t2_loc[frame][1], '.')
    ax.plot(t3_loc[frame][0], t3_loc[frame][1], '.')
    ax.plot(t4_loc[frame][0], t4_loc[frame][1], '.')

    if frame not in discarded_frame_set:
        #ax.plot(spline_points[frame][:, 0], spline_points[frame][:, 1], 'g-')
        #ax.plot(spline_points_rev[frame][:, 0], spline_points_rev[frame][:, 1], 'm-')
        ax.plot(spline_avg[frame][:, 0], spline_avg[frame][:, 1], 'y-')
    else:
        ax.plot(spline_avg[frame][:, 0], spline_avg[frame][:, 1], 'r,-')
        circle = Circle((200, 200), 10, color='red', fill=True, linewidth=2)
        ax.add_patch(circle)

    ax.set_title(f'frame={frame}, score:{str(round(instance_scores[0][frame], 1))}')
    ax.axis('off')
    
def plot_random_frames_test_spline_avg(len_rec, myVideo, spline_avg, t1_loc, head_loc, t2_loc, t3_loc, t4_loc, discarded_frame_set,instance_scores):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8), dpi=100)
    frames = random.sample(range(len_rec), 9)
    for ax, frame in zip(axes.flatten(), frames):
        plot_test_spline(ax, frame, myVideo, spline_avg, t1_loc, head_loc, t2_loc, t3_loc, t4_loc, discarded_frame_set,instance_scores)
    plt.tight_layout()
    plt.show()

def plot_test_final(ax, frame, myVideo, spline_sy0, com_sy0, discarded_frame_set, instance_scores, orient):
    ax.imshow(myVideo[:,:,frame], origin='lower', cmap='gray')
    if frame not in discarded_frame_set:
        ax.plot(spline_sy0[frame][:, 0], spline_sy0[frame][:, 1], 'g-')
        ax.plot(com_sy0[frame][0], com_sy0[frame][1], 'x')
        angle_radians = np.radians(orient[frame])
        endpoint = com_sy0[frame] - 50 * np.array([np.cos(angle_radians), np.sin(angle_radians)])
        ax.plot([com_sy0[frame][0], endpoint[0]], [com_sy0[frame][1], endpoint[1]], marker='.')
    
    else:
        circle = Circle((200, 200), 10, color='red', fill=True, linewidth=2)
        ax.add_patch(circle)

    ax.set_title(f'frame={frame}, score:{str(round(instance_scores[0][frame], 1))}')
    ax.axis('off')
    
def plot_random_frames_test_final(len_rec, myVideo, spline_sy0, com_sy0, discarded_frame_set, instance_scores, orient):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8), dpi=100)
    frames = random.sample(range(len_rec), 9)
    for ax, frame in zip(axes.flatten(), frames):
        plot_test_final(ax, frame, myVideo, spline_sy0, com_sy0, discarded_frame_set, instance_scores, orient)
    plt.tight_layout()
    plt.show()
    
    
def reverse_colourmap(cmap, name = 'my_cmap_r'):
    """
    In: 
    cmap, name 
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """        
    reverse = []
    k = []   

    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL) 
    return my_cmap_r

def fig_traj_allRuns(results, time_runs, category_names, title):
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['cividis'](
        np.linspace(0.15, 0.85, data.shape[1])) #['RdYlGn']

    fig, ax = plt.subplots(2,1,figsize=(5, 5),gridspec_kw={'height_ratios': [1, 6]})
    ax[0].invert_yaxis()
    ax[0].xaxis.set_visible(False)
    ax[0].set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax[0].barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax[0].bar_label(rects, label_type='center', color=text_color)
        ax[0].set_title(title)
    # Adding the text on the right side
    #for i in range(len(time_runs)):
    #    ax[0].text(np.sum(data, axis=1).max(), i, time_runs[i], va='center')

    return fig, ax, category_colors