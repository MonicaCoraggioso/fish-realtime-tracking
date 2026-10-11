"""
    Background to project below the entire tank during freely experiment.
    
    Author: Monica Coraggioso
"""
from psychopy import visual,event,clock,monitors,core
from PIL import Image
import os
import numpy as np
import cv2
import random

def stones_bg_proj(path, name_img, win):
    """Define the background to project below the entire tank
    
    Parameters
    ----------
    path : image path
    name_img : imgage name (e.g. 'bg-grayscale.jpg')
    win : projector window already opened with psychopy
    
    Returns
    -------
    img_static = visual.ImageStim(...)
    """
    im = Image.open(os.path.join(path,name_img))
    # Convert the image to a NumPy array
    im_array = np.array(im)
    
    # Rotate the NumPy array (image) by 180 degrees
    rotated_im_array = cv2.rotate(im_array, cv2.ROTATE_90_CLOCKWISE)
    
    # Convert the rotated NumPy array back to a Pillow image
    im = Image.fromarray(rotated_im_array)
    
    img_static = visual.ImageStim(win, image=im, pos=(-3,+1), size=(44,45), units="cm", colorSpace="rgb1")

    im_pos=img_static.pos
    im_size=img_static.size
    return img_static, rotated_im_array, im_pos, im_size

def stones_bg_proj_RandomAngle(path, name_img, win):
    """Define the background to project below the entire tank
    
    Parameters
    ----------
    path : image path
    name_img : imgage name (e.g. 'bg-grayscale.jpg')
    win : projector window already opened with psychopy
    
    Returns
    -------
    img_static = visual.ImageStim(...)
    """
    im = Image.open(os.path.join(path,name_img))
    angle = random.choice([0, 90, 180, 270])
    rotated_im = im.rotate(angle, expand=True)

    # Convert the image to a NumPy array
    im_array = np.array(rotated_im)
    
    # Rotate the NumPy array (image) by 180 degrees
    rotated_im_array = cv2.rotate(im_array, cv2.ROTATE_90_CLOCKWISE)
    
    # Convert the rotated NumPy array back to a Pillow image
    im = Image.fromarray(rotated_im_array)
    
    img_static =visual.ImageStim(win, image=im, pos=(0,0), size=(40,45), units="cm", colorSpace="rgb1")
    im_pos=img_static.pos
    im_size=img_static.size
    
    return img_static, rotated_im_array, im_pos, im_size

def pad_to_square_pow2(tex):
    h, w = tex.shape[:2]
    new_size = 2 ** int(np.ceil(np.log2(max(h, w))))
    
    if tex.ndim == 2:  # grayscale
        padded = np.zeros((new_size, new_size), dtype=tex.dtype)
        padded[:h, :w] = tex
    elif tex.ndim == 3:  # RGB
        padded = np.zeros((new_size, new_size, tex.shape[2]), dtype=tex.dtype)
        padded[:h, :w, :] = tex
    else:
        raise ValueError("Unsupported texture shape: {}".format(tex.shape))
    
    return padded
    
def stones_bg_screen(path, name_img, win):
    im = Image.open(os.path.join(path, name_img))
    im_array = np.array(im)  # Convert the image to a NumPy array
    texture = (im_array / 255.0) * 2 - 1
    try:
        image_height, image_width, _ = im_array.shape
    except:
        image_height, image_width = im_array.shape
    texture = pad_to_square_pow2(texture)
    img_static = visual.GratingStim(win, tex=texture, pos=(0, 0), size=(image_width*2.2, image_height*2.2), interpolate=True, colorSpace='rgb', units='pix')
    im_pos=img_static.pos
    im_size=img_static.size
    return img_static, im_array, im_pos, im_size

def stones_bg_screen_4moving_patter(path, name_img, win):
    im = Image.open(os.path.join(path, name_img))
    im_array = np.array(im)  # Convert the image to a NumPy array
    texture = (im_array / 255.0) * 2 - 1
    try:
        image_height, image_width, _ = im_array.shape
    except:
        image_height, image_width = im_array.shape
    texture = pad_to_square_pow2(texture)
    img_static = visual.GratingStim(win, tex=texture, pos=(0, 0), size=(image_width*3, image_height*3), interpolate=True, colorSpace='rgb', units='pix')
    im_pos=img_static.pos
    im_size=img_static.size
    return img_static, im_array, im_pos, im_size

def closeloop_bg_screen(path, name_img, win):
    im = Image.open(os.path.join(path, name_img))
    im_array = np.array(im)  # Convert the image to a NumPy array
    texture = (im_array / 255.0) * 2 - 1
    try:
        image_height, image_width, _ = im_array.shape
    except:
        image_height, image_width = im_array.shape

    win_size_px = win.size
    # Create the background image
    screen_width_cm = 75  # Example: width of screen in cm
    screen_height_cm = 46  # Example: height of screen in cm
    screen_width_px = 2560  # Example: screen resolution width in pixels
    screen_height_px = 1440  # Example: screen resolution height in pixels
    
    # Conversion factors
    cm_per_px_width = screen_width_cm / screen_width_px
    cm_per_px_height = screen_height_cm / screen_height_px
    
    # Compute window size in cm
    win_width_cm = win_size_px[0] * cm_per_px_width
    win_height_cm = win_size_px[1] * cm_per_px_height
    
    # Create GratingStim with the updated size
    img_static = visual.GratingStim(
        win,
        tex=texture,
        pos=(0, 0),
        sf=1/40,#1/25,
        size=(win_width_cm*2, win_height_cm*2),  # Set size to match window dimensions
        interpolate=True,
        colorSpace='rgb',
        units='cm'
    )
    im_pos = img_static.pos
    im_size = img_static.size

    # Draw the background and the cross
    img_static.draw()
    win.flip()

    return img_static, im_array, im_pos, im_size

def stones_bg_screen_framed(path, name_img, win, frame_gradient):
    """Define the background to project below the entire tank: 
    in this version you have a frame gradient to correct the difference in illumination due to the projector.

    The gradient frame is first created with a gradient calibrated to have a consistent flat illumination, then multiplied with the stones_img.
    
    Parameters
    ----------
    path : image path
    name_img : image name (e.g. 'bg-grayscale.jpg')
    win : projector window already opened with psychopy
    
    Returns
    -------
    img_static = visual.ImageStim(...)
    """
    im = Image.open(os.path.join(path, name_img))
    im_array = np.array(im)  # Convert the image to a NumPy array
    if frame_gradient:
        width, height = 910, 645  # To be consistent with the img_stones.size
        alpha = 2  # calibrated 06.06.24
        
        # Assuming 1 cm corresponds to a certain number of pixels
        pixels_per_cm = 150  # This is an example value; you need to calibrate this  
        frame_width = pixels_per_cm  # 1 cm frame width
        
        # Create a gradient frame
        gradient = np.ones((height, width), dtype=np.float32)
        
        # Create a bright frame of 1 cm width around the edges
        for i in range(height):
            for j in range(width):
                distance_to_edge = min(i, j, height - 1 - i, width - 1 - j)
                if distance_to_edge < frame_width:
                    gradient[i, j] += alpha * (frame_width - distance_to_edge) / frame_width
        
        gradient = np.clip(gradient, 1, 1 + alpha)  # Ensure the values are within [1, 1 + alpha]
        gradient = (gradient * 255 / (1 + alpha)).astype(np.uint8)  # Convert to 0-255 range, scaled by alpha
        
        # Ensure there are no zero values in gradient_image
        gradient[gradient == 0] = 255  # Replace 0 values with 255 (equivalent to a normalized value of 1)
        
        # Convert to RGB image
        gradient_image = np.stack((gradient, gradient, gradient), axis=2)
        
        rotated_im_array = (im_array * (gradient_image / 255)).astype(np.uint8)  # stones * normalized(gradient)
    else:
        rotated_im_array = im_array.astype(np.uint8)
    
    # Convert the rotated NumPy array back to a Pillow image
    im = Image.fromarray(rotated_im_array)
    
    #img_static = visual.ImageStim(win, image=im, pos=(-3, 0), size=(45, 45), units="cm", colorSpace="rgb1") 
    texture = im_array / 255.0  # Normalize the image array to be between 0 and 1
    img_static = visual.GratingStim(win, tex=texture, pos=(-3, 0), size=(45, 45), units="cm", colorSpace="rgb1")

    im_pos=img_static.pos
    im_size=img_static.size
    
    return img_static, rotated_im_array, im_pos, im_size

