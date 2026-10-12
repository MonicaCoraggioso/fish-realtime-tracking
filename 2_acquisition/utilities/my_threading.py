"""
    Thredind: 3 thread for paralel computation:
    -Camera: running at around 200Hz + tracking position and tail at around 130Hz. Only the position is shared in real time with the motor
    -Motor: running at around 20Hz, it receaves in real ime the position of the fish from the camera thread and make a move in order to have the fish always in the center of the camera ROI
    -Save images: only if save_flag=True
    
    Author: Monica Coraggioso
"""
import numpy as np
import threading
import queue
import time
from tracking import centroid, get_ellipse, img_draw_ellipse, trackng_function
from camera_PointGrey import _initialize, _cleanup, capture


#OLD VERSION: save movie in .npy array
def run_saving_img(queue_images,stop_saving,dir_experiment_fish):
    if not os.path.exists(os.path.join(dir_experiment_fish,'movie')):
        os.mkdir(os.path.join(dir_experiment_fish,'movie'))
    Delta_save=int(250*1*30)#video crop in ~1minutes
    x_movie, y_movie, t_movie = img_crop_ds.shape[0], img_crop_ds.shape[1], Delta_save
    movie_0=np.empty((x_movie, y_movie, t_movie),dtype='uint8')
    movie_1=np.empty((x_movie, y_movie, t_movie),dtype='uint8')
    movie_id_pari=True
    movie_id=0
    
    while not (stop_saving.is_set() and queue_images.empty()):
        if not queue_images.empty():
            frame,frame_id=queue_images.get()    
            frame_id=frame_id-movie_id*Delta_save
            if movie_id_pari:
                movie_0[:,:,frame_id]=frame
                if frame_id % (Delta_save-1) == 0 and frame_id!=0:
                    movie_id_pari=not movie_id_pari
                    np.save(os.path.join(dir_experiment_fish,'movie',f'movie_{movie_id}'),movie_0)    
                    movie_id+=1
            else:
                movie_1[:,:,frame_id]=frame
                if frame_id % (Delta_save-1) == 0 and frame_id!=0:
                    movie_id_pari=not movie_id_pari
                    np.save(os.path.join(dir_experiment_fish,'movie',f'movie_{movie_id}'),movie_1)    
                    movie_id+=1
        time.sleep(0.001) 