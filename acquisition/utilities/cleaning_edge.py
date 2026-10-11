"""
    Functions to remove any edge residue after applying the mask
    
    Author: Monica Coraggioso
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import downscale_local_mean
import cv2
from scipy.signal import find_peaks, peak_widths
from utilities.preprocessing import noise_removal

def residual_edge_removal(image, plot=None):
    #timeit: 527 µs ± 8 µs per loop
    """Function that removes any edge residue after applying the mask. 
    In fact, as a result of backlush, information about the position of the motor acquires noise over the course of the     experiment and thus the mask becomes misaligned to the camera ROI.
    The idea is to remove any lines due to the edge by integrating the intensities along the x-axis and y-axis. 
    If the integral has a pronounced peak it means that the line is present and the corresponding portion is set to zero     in the binarized image. 
  
    Parameters
    ----------
    image: final binarized image, already preprocessed and mupltiplied by 'mask'
    plot: True if you want to plot 
    
    Returns
    ----------
    image: binarized image with 0 at a residual edge line
    """
    kernel = np.ones((3,3), np.uint8)
    
    row_intensities = np.sum(image, axis=1)
    p_r, _ = find_peaks(row_intensities,prominence=1.5,height=100)
    if len(p_r)!=0:
        max_p_r = np.argmax(row_intensities[p_r])
        if row_intensities[p_r[max_p_r]]>100:
            results_r = peak_widths(row_intensities, p_r, rel_height=0.7)
            aa=int(results_r[2][0]), int(results_r[1][0])
            bb=int(results_r[3][0]), int(results_r[1][0])
            image[aa[0]:bb[0]]=0
            image = cv2.erode(image, kernel)

    col_intensities = np.sum(image, axis=0)
    p_c, _ = find_peaks(col_intensities,prominence=1.5,height=100)
    if len(p_c)!=0:
        max_p_c = np.argmax(col_intensities[p_c])
        if col_intensities[p_c[max_p_c]]>100:
            results_c = peak_widths(col_intensities, p_c, rel_height=0.7)
            cc=int(results_c[2][0]), int(results_c[1][0])
            dd=int(results_c[3][0]), int(results_c[1][0])
            image[:,cc[0]:dd[0]]=0         
            image = cv2.erode(image, kernel)
            
    if plot!=None:
        fig,axes=plt.subplots(1,2,figsize=(8,3),sharey=True)
        ax=axes.ravel()
        ax[0].plot(row_intensities, label='row_intensities')
        if len(p_r)!=0:
            ax[0].plot([p_r[max_p_r]], row_intensities[p_r[max_p_r]], 'ro', label='peak')
            ax[0].plot(aa[0],aa[1], 'go', label='left_bases')
            ax[0].plot(bb[0],bb[1], 'yo', label='right_bases')
        ax[0].legend()
        ax[1].plot(col_intensities, label='col_intensities')
        if len(p_c)!=0:
            ax[1].plot([p_c[max_p_c]], col_intensities[p_c[max_p_c]], 'ro', label='peak')
            ax[1].plot(cc[0],cc[1], 'go', label='left_bases')
            ax[1].plot(dd[0],dd[1], 'yo', label='right_bases')
        ax[1].legend()
    return image


def clear_res_border(img, area, b_left, b_right, b_up, b_down):
    #timeit: 4 ms per loop
    """Integrate the image along the col/row parallel to the edge until reach area=area_fish, put at zero the rest. 
    
    Parameters:
    ----------
    img = bw_image
    area = area_fish in pix
    b_left = boolean, left border in the image
    b_right = boolean, right border in the image
    b_up = boolean, up border in the image
    b_down = boolean, down border in the image
    
    Inside:
    --------- 
    img = bw_image cleaned
    """
    pix_sum = np.array([0])
    if b_left:
        for col in range(img.shape[0]):
            if not col==0:
                opp_col=img.shape[0]-col
                pix_sum=np.concatenate((np.array([ np.sum(img[:,opp_col]) ]), pix_sum)) 
                pix_area=np.sum(pix_sum)
                if pix_area>=area:
                    img[:,0:opp_col]=0
                    break

    if b_right:
        for col in range(img.shape[0]):
            if not col==0:
                pix_sum=np.concatenate((np.array([ np.sum(img[:,col]) ]), pix_sum)) 
                pix_area=np.sum(pix_sum)
                if pix_area>=area:
                    img[:,col:]=0
                    break  

    if b_up:
        for row in range(img.shape[1]):
            if not row==0:
                pix_sum=np.concatenate((np.array([ np.sum( img[row,:]) ]), pix_sum)) 
                pix_area=np.sum(pix_sum)
                if pix_area>=area:
                    img[row:,:]=0
                    break  

    if b_down:
        for row in range(img.shape[1]):
            if not row==0:
                opp_row=img.shape[1]-row
                pix_sum=np.concatenate((np.array([ np.sum(img[opp_row,:]) ]), pix_sum)) 
                pix_area=np.sum(pix_sum)
                if pix_area>=area:
                    img[0:opp_row,:]=0
                    break  
                    
    img_c=noise_removal(img, 100)
    return img_c