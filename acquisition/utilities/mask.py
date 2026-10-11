"""
    Online tracking: position of the fish + tail points
    
    Author: Monica Coraggioso
"""
import numpy as np
import math
import cv2
from scipy.signal import find_peaks
import re #regex
import serial, time #pySerial
from utilities.camera_PointGrey import _initialize, _cleanup, capture
from utilities.preprocessing import preprocess_im, bw_image, noise_removal, area_contours
from utilities.Shapeoko import ShapeokoStatus, ShapeokoWrite, ShapeokoMove
from scipy import ndimage

def find_edges(image, h, d):
    
    m=np.sum(image,axis=0).astype("float") # m=colum
    l=np.sum(image,axis=1).astype("float") # l=row
    
    dm=np.abs(np.diff(m))
    dl=np.abs(np.diff(l))
    
    peaks_dm, prop_dm = find_peaks(dm, height=h, distance=d)
    peaks_dl, prop_dl = find_peaks(dl, height=h, distance=d)
    heights_dm = prop_dm['peak_heights']
    heights_dl = prop_dl['peak_heights']
    
    return m, l, dm, dl, peaks_dm, peaks_dl, heights_dm, heights_dl

def take_mask(mask, camera, gaussian_sigma, downsampling, background, th_img, noise_removal_threshold, x_motor_pix, y_motor_pix, h_img, w_img):    
    #take image and preprocess it exactly as during the experiment
    _initialize(camera)
    [img_raw,timestamp]=capture(camera) 
    _cleanup(camera)
    [img_bw0, img_bg, img_ds]=bw_image(img_raw, gaussian_sigma, downsampling, background, th_img)
    img_bw=1-noise_removal(img_bw0, noise_removal_threshold) 
  
    #place it inside the mask
    m_tmp=mask[x_motor_pix:x_motor_pix+h_img, y_motor_pix:y_motor_pix+w_img]
    mask[x_motor_pix:x_motor_pix+h_img, y_motor_pix:y_motor_pix+w_img]=np.multiply(img_bw.T,m_tmp)
    
    return mask

from utilities.saving import cropImage
from skimage.transform import downscale_local_mean
from utilities.preprocessing import background_sub

def take_mask_2(mask, mot_xB,mot_yC, camera, downsampling, bg_raw_crop_ds, two_thresholds, lower_threshold, upper_threshold, unique_threshold, noise_removal_threshold, x_motor_pix, y_motor_pix, h_img, w_img):
    """Function that take an image along the border, do the EXACT (offline version) preprocess that during the experiment and use the complementary final image (0=obj,1=not obj). 
    """
    if camera.is_streaming():
        _cleanup(camera)
    _initialize(camera)
    [img_raw,timestamp]=capture(camera) 
    _cleanup(camera)
    img_crop=cropImage(img_raw, 300 , 300, 500)
    img_crop_ds=downscale_local_mean(img_crop,downsampling)
    img_crop_ds_bg=background_sub(img_crop_ds,bg_raw_crop_ds) 
    if two_thresholds:
        ret, threshold1 = cv2.threshold(img_crop_ds_bg, lower_threshold, 1, cv2.THRESH_BINARY)
        ret, threshold2 = cv2.threshold(img_crop_ds_bg, upper_threshold, 1, cv2.THRESH_BINARY_INV)
        img_bw0 = cv2.bitwise_and(threshold1, threshold2)
    else:
        img_bw0 = cv2.threshold(img_crop_ds_bg, unique_threshold, 1, cv2.THRESH_BINARY)[1]
    img_bw=1-noise_removal(img_bw0, noise_removal_threshold)
  
    ##place it inside the mask-old
    #if x_motor_pix<mot_xB/2:
    #    sx,ex=x_motor_pix,x_motor_pix+h_img
    #else:
    #    sx,ex=x_motor_pix-h_img,x_motor_pix  
    #if y_motor_pix<mot_yC/2:
    #    sy,ey=y_motor_pix,y_motor_pix+w_img
    #else:
    #    sy,ey=y_motor_pix-w_img,y_motor_pix
    #m_tmp=mask[sx:ex, sy:ey] 
    #mask[sx:ex, sy:ey]=np.multiply(img_bw.T,m_tmp)
    #place it inside the mask
    m_tmp=mask[y_motor_pix:y_motor_pix+w_img,x_motor_pix:x_motor_pix+h_img]
    mask[y_motor_pix:y_motor_pix+w_img,x_motor_pix:x_motor_pix+h_img]=np.multiply(img_bw,m_tmp)
    
    return mask

def mask_status_report(ser,mm_per_pix):
    time.sleep(0.1) 
    msg_status=ShapeokoWrite(ser,'?',False)
    x_motor,y_motor,x_motor_pix,y_motor_pix=np.nan,np.nan,np.nan,np.nan
    while '<Run' in msg_status:
        time.sleep(0.1) 
        msg_status=ShapeokoWrite(ser,'?',False)
        if '<Idle' in msg_status:
            if 'WPos:' in msg_status:
                s0=re.split('[|]',msg_status) 
                wpos=re.split('[:,]',s0[1])  
                x_motor,y_motor=float(wpos[1]),float(wpos[2])
                x_motor_pix,y_motor_pix=int(x_motor/mm_per_pix),int(y_motor/mm_per_pix)
            else:
                print('ERROR:',msg_status)  
    
    return x_motor,y_motor,x_motor_pix,y_motor_pix


def mask_status_report_2(ser,mm_per_pix):
    time.sleep(0.1) 
    msg_status=ShapeokoWrite(ser,'?',False)
    x_motor,y_motor,x_motor_pix,y_motor_pix=np.nan,np.nan,np.nan,np.nan
    msg_status=ShapeokoWrite(ser,'?',False)
    if '<Idle' in msg_status:
        if 'WPos:' in msg_status:
            s0=re.split('[|]',msg_status) 
            wpos=re.split('[:,]',s0[1])  
            x_motor,y_motor=float(wpos[1]),float(wpos[2])
            x_motor_pix,y_motor_pix=int(x_motor/mm_per_pix),int(y_motor/mm_per_pix)
        else:
            print('ERROR:',msg_status)  
    
    return x_motor,y_motor,x_motor_pix,y_motor_pix


def mask_creation(masks,s,ser,mm_per_pix,mot_xA,mot_yA,mot_xB,mot_yB,mot_xC,mot_yC,mot_xD,mot_yD,camera, downsampling, bg_raw_crop_ds, two_thresholds, lower_threshold, upper_threshold, unique_threshold, noise_removal_threshold, x_motor_pix, y_motor_pix, h_img, w_img):
    """Function that create a mask using the image from 'take_mask' function.
   
    Parameters
    ----------
    masks : initial matrix to be filled. More then one if you want moltiple masks.
    s : motor step
    mot_xA,mot_yA,mot_xB,mot_yB,mot_xC,mot_yC,mot_xD,mot_yD : edge boundaries
    
    Returns
    ----------
    masks
    """
    for id_mask in range(len(masks)):    
        mask=masks[id_mask]
        t0=time.time()
        ser.write(b'G90')

        print('segmentAB')
        ShapeokoMove(ser,mot_xA,mot_yA,3000)
        x_motor,y_motor=mot_xA,mot_yA
        x_motor_pix,y_motor_pix=int(x_motor/mm_per_pix),int(y_motor/mm_per_pix)
        time.sleep(1)
        while x_motor<=mot_xB-s:
            mask=take_mask_2(mask, camera, downsampling, bg_raw_crop_ds, two_thresholds, lower_threshold, upper_threshold, unique_threshold, noise_removal_threshold, x_motor_pix, y_motor_pix, h_img, w_img)
            ShapeokoMove(ser,x_motor+s,mot_yA,3000)
            time.sleep(0.1)
            [x_motor,y_motor,x_motor_pix,y_motor_pix]=mask_status_report(ser,mm_per_pix)
        mask=take_mask_2(mask, camera, downsampling, bg_mean, two_thresholds, lower_threshold, upper_threshold, unique_threshold, noise_removal_threshold_m, x_motor_pix, y_motor_pix, h_img, w_img)
            
        print('segmentBC')
        ShapeokoMove(ser,mot_xB,mot_yB,3000)
        x_motor,y_motor=mot_xB,mot_yB
        x_motor_pix,y_motor_pix=int(x_motor/mm_per_pix),int(y_motor/mm_per_pix)
        time.sleep(1)
        while y_motor<=mot_yC-s:
            mask=take_mask_2(mask, camera, downsampling, bg_raw_crop_ds, two_thresholds, lower_threshold, upper_threshold, unique_threshold, noise_removal_threshold, x_motor_pix, y_motor_pix, h_img, w_img)
            ShapeokoMove(ser,mot_xB,y_motor+s,3000)
            time.sleep(0.1)
            [x_motor,y_motor,x_motor_pix,y_motor_pix]=mask_status_report(ser,mm_per_pix)
        mask=take_mask_2(mask, camera, downsampling, bg_mean, two_thresholds, lower_threshold, upper_threshold, unique_threshold, noise_removal_threshold_m, x_motor_pix, y_motor_pix, h_img, w_img)
            
        print('segmentDC')
        ShapeokoMove(ser,mot_xD,mot_yD,3000)
        x_motor,y_motor=mot_xD,mot_yD
        x_motor_pix,y_motor_pix=int(x_motor/mm_per_pix),int(y_motor/mm_per_pix)
        time.sleep(1)
        while x_motor<=mot_xC-s:
            mask=take_mask_2(mask, camera, downsampling, bg_raw_crop_ds, two_thresholds, lower_threshold, upper_threshold, unique_threshold, noise_removal_threshold, x_motor_pix, y_motor_pix, h_img, w_img)
            ShapeokoMove(ser,x_motor+s,mot_yD,3000)
            time.sleep(0.1)
            [x_motor,y_motor,x_motor_pix,y_motor_pix]=mask_status_report(ser,mm_per_pix)
        mask=take_mask_2(mask, camera, downsampling, bg_mean, two_thresholds, lower_threshold, upper_threshold, unique_threshold, noise_removal_threshold_m, x_motor_pix, y_motor_pix, h_img, w_img)
            
        print('segmentAD')
        ShapeokoMove(ser,mot_xA,mot_yA,3000)
        x_motor,y_motor=mot_xA,mot_yA
        x_motor_pix,y_motor_pix=int(x_motor/mm_per_pix),int(y_motor/mm_per_pix)
        time.sleep(1)
        while y_motor<=mot_yD-s:
            mask=take_mask_2(mask, camera, downsampling, bg_raw_crop_ds, two_thresholds, lower_threshold, upper_threshold, unique_threshold, noise_removal_threshold, x_motor_pix, y_motor_pix, h_img, w_img)
            ShapeokoMove(ser,mot_xA,y_motor+s,3000)
            time.sleep(0.1)
            [x_motor,y_motor,x_motor_pix,y_motor_pix]=mask_status_report(ser,mm_per_pix)
        mask=take_mask_2(mask, camera, downsampling, bg_mean, two_thresholds, lower_threshold, upper_threshold, unique_threshold, noise_removal_threshold_m, x_motor_pix, y_motor_pix, h_img, w_img)
            
        masks[id_mask]=mask.astype(np.uint8)
        time.sleep(1)
        print('Mask done in', int(time.time()-t0), 's') 
        
    return masks[id_mask]

import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
def visualize_mask(ser,xp,yp, mm_per_pix, camera, bg_mean, downsampling, two_thresholds, 
                   lower_threshold, upper_threshold, unique_threshold, noise_removal_threshold,mask, h_img, w_img):
    time.sleep(1)
    msg_status=ShapeokoWrite(ser,'?',True)
           
    if 'WPos:' in msg_status:
        s0=re.split('[|]',msg_status) 
        wpos=re.split('[:,]',s0[1])  
        buffer=re.split('[:,]',s0[2])  
    x_motor_pix,y_motor_pix=int(float(wpos[1])/mm_per_pix),int(float(wpos[2])/mm_per_pix)
    print(x_motor_pix,y_motor_pix)
    
    _initialize(camera)
    [img_raw,timestamp]=capture(camera) 
    _cleanup(camera)
    img_crop=cropImage(img_raw, 300 , 300, 500)
    img_crop_ds=downscale_local_mean(img_crop,downsampling)
    img_crop_ds_bg=background_sub(img_crop_ds,bg_mean) 
    if two_thresholds:
        ret, threshold1 = cv2.threshold(img_crop_ds_bg, lower_threshold, 1, cv2.THRESH_BINARY)
        ret, threshold2 = cv2.threshold(img_crop_ds_bg, upper_threshold, 1, cv2.THRESH_BINARY_INV)
        img_bw0 = cv2.bitwise_and(threshold1, threshold2)
    else:
        img_bw0 = cv2.threshold(img_crop_ds_bg, unique_threshold, 1, cv2.THRESH_BINARY)[1]
    
    mask=mask.astype(np.uint8)
    m_tmp=mask[y_motor_pix:y_motor_pix+w_img,x_motor_pix:x_motor_pix+h_img] 
    img_bw=np.multiply(img_bw0,m_tmp)
    img_bw_clean=noise_removal(img_bw, noise_removal_threshold)  
    
    print(area_contours(img_bw_clean))
    fig, axes = plt.subplots(3,2, figsize=(5, 5),sharex=True,sharey=True)
    ax = axes.ravel()  
    titles = ['img_raw', 'img_bw0', 'm_tmp', 'img_bw0-m_tmp', 'm_tmp-img_bw0', 'img_bw']
    images = [img_crop_ds_bg, img_bw0, m_tmp, img_bw0-m_tmp, m_tmp-img_bw0, img_bw]
    for i in range(6):
        plt.subplot(3,2,i+1),plt.imshow(images[i],origin='lower')
        plt.title(titles[i])
        plt.grid(which='minor')
        plt.plot(xp,yp,'r.')
        #plt.xticks([]),plt.yticks([])
    plt.show()
    shift, error, diffphase = phase_cross_correlation(m_tmp, img_bw0)
    return x_motor_pix,y_motor_pix,shift


