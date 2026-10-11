"""
    Set of functions that allow to project a light pattern in the center of the camera, i.e. where the fish is located.
    
    Author: Monica Coraggioso
"""
import numpy as np
from psychopy import visual,event,clock,monitors,core
import serial, time #pySerial
import re #regex
from utilities.Shapeoko import ShapeokoMove, ShapeokoStatus, ShapeokoWrite

    
def manual_mode_projector(ser, x_center_motor, y_center_motor, win, dl, img_static=None):
    """Manually guide the motor with the arrows on the keyboard. 
    The goal is to bring the fish into the center of view of the camera. 
    For helping purpose, it is projected a point in the center of the camera and a square that corresponds to the camera-ROI.

    Parameters
    ----------
    ser : serial comunication with Shapeoko
    x_center_motor , y_motor_center : initial position of the motor that correspond to (0,0) in the coordinate of the projector (calibrated in 'calibrations.ipynb)
    win : visual.Window already open in the main
    dl : lenght of the manual step [mm]

    Note
    ----------
    Press 's' to exit. Different from 'manual_mode' because here the 'win' is open OUTSIDE the function

    """
    ser.write(b'G91\n')
    time.sleep(0.01)
    #print('Relative mode')
    
    x_motor=[]
    y_motor=[]

    time.sleep(0.01)#print('Relative mode')
    msg_status=ShapeokoWrite(ser,'?',False)
    if 'WPos:' in msg_status:
        s0=re.split('[|]',msg_status) 
        wpos=re.split('[:,]',s0[1])  
        x_motor_now=float(wpos[1])
        y_motor_now=float(wpos[2])
        x_motor.append(x_motor_now)
        y_motor.append(y_motor_now)
    else:
        print('Error')
    x_alig=(x_center_motor-x_motor_now)/10 
    y_alig=(y_center_motor-y_motor_now)/10

    ref_pos=np.array([x_alig, y_alig])
    circle=visual.Circle(win=win, radius=1, fillColor='gray', pos=ref_pos)
    blank=visual.Rect(win=win, width=60, height=60, fillColor='gray', pos=(0,0))
    
    v_step=10000
    flag_blank=False
    while True: 
        time.sleep(0.01)#print('Relative mode')
        msg_status=ShapeokoWrite(ser,'?',False)
        if 'WPos:' in msg_status:
            s0=re.split('[|]',msg_status) 
            wpos=re.split('[:,]',s0[1])  
            x_motor_now=float(wpos[1])
            y_motor_now=float(wpos[2])
            x_motor.append(x_motor_now)
            y_motor.append(y_motor_now)
        else:
            print('Error')
        dx_motor=(x_motor[-1]-x_motor[-2])/10
        dy_motor=(y_motor[-1]-y_motor[-2])/10
        
        if img_static!=None:
            img_static.draw()
 
        if flag_blank:
            blank.draw()
        circle.draw()
        win.flip()
        
        key=event.getKeys()
            
        if key==['s']:
            img_static.draw()
            win.flip() 
            ser.write(b'G90\n')
            time.sleep(0.01)
            print('End manual mode...')
            break

        if key==['up']:
            ShapeokoMove(ser,-dl,0)
            ref_pos=[ref_pos[0]+dy_motor,ref_pos[1]-dx_motor]
            circle.pos=ref_pos
        if key==['down']:
            ShapeokoMove(ser,dl,0)
            ref_pos=[ref_pos[0]+dy_motor,ref_pos[1]-dx_motor]
            circle.pos=ref_pos
        if key==['right']:
            ShapeokoMove(ser,0,dl)
            ref_pos=[ref_pos[0]+dy_motor,ref_pos[1]-dx_motor]
            circle.pos=ref_pos
        if key==['left']:
            ShapeokoMove(ser,0,-dl)
            ref_pos=[ref_pos[0]+dy_motor,ref_pos[1]-dx_motor]
            circle.pos=ref_pos
            
        if key==['q']:
            flag_blank = not flag_blank       
        event.clearEvents()

def manual_mode_screen(ser, x_center_motor, y_center_motor, win, dl, img_static=None):
    """Manually guide the motor with the arrows on the keyboard. 
    The goal is to bring the fish into the center of view of the camera. 
    For helping purpose, it is projected a point in the center of the camera and a square that corresponds to the camera-ROI.

    Parameters
    ----------
    ser : serial comunication with Shapeoko
    x_center_motor , y_motor_center : initial position of the motor that correspond to (0,0) in the coordinate of the projector (calibrated in 'calibrations.ipynb)
    win : visual.Window already open in the main
    dl : lenght of the manual step [mm]

    Note
    ----------
    Press 's' to exit. Different from 'manual_mode' because here the 'win' is open OUTSIDE the function

    """
    ser.write(b'G91\n')
    time.sleep(0.01)
    #print('Relative mode')
    
    msg_status=ShapeokoWrite(ser,'?',True)
    if 'WPos:' in msg_status:
        s0=re.split('[|]',msg_status) 
        wpos=re.split('[:,]',s0[1])  
        x_motor_now=float(wpos[1])
        y_motor_now=float(wpos[2])
    else:
        print('Error')

    #to align the ref_pos of the projector at the camera center of view, taking into account the different axis and the fact that the motor work in [mm] and the projector in [cm]
    x_alig=(x_center_motor-x_motor_now)/10 
    y_alig=(y_center_motor-y_motor_now)/10

    ref_pos=np.array([x_alig, y_alig])
    circle=visual.Circle(win=win, radius=0.1, fillColor='gray', pos=ref_pos)
    rect=visual.Rect(win=win, width=1.5, height=1.5, lineColor='black', fillColor='gray', pos=ref_pos)
    blank=visual.Rect(win=win, width=60, height=60, fillColor='gray', pos=(0,0))
    
    v_step=10000
    flag_blank=False
    while True: 
        
        if img_static!=None:
            img_static.draw()
 
        if flag_blank:
            blank.draw()
        rect.draw()
        circle.draw()
        win.flip()
        
        key=event.getKeys()
            
        if key==['s']:
            img_static.draw()
            win.flip() 
            ser.write(b'G90\n')
            time.sleep(0.01)
            print('End manual mode...')
            break

        if key==['up']:
            ShapeokoMove(ser,-dl,0,v_step)
            ref_pos=[ref_pos[0],ref_pos[1]+dl/10]
            rect.pos=ref_pos
            circle.pos=ref_pos
        if key==['down']:
            ShapeokoMove(ser,dl,0,v_step)
            ref_pos=[ref_pos[0],ref_pos[1]-dl/10]
            rect.pos=ref_pos
            circle.pos=ref_pos
        if key==['right']:
            ShapeokoMove(ser,0,dl,v_step)
            ref_pos=[ref_pos[0]+dl/10,ref_pos[1]]
            rect.pos=ref_pos
            circle.pos=ref_pos
        if key==['left']:
            ShapeokoMove(ser,0,-dl,v_step)
            ref_pos=[ref_pos[0]-dl/10,ref_pos[1]]
            rect.pos=ref_pos
            circle.pos=ref_pos
            
        if key==['q']:
            flag_blank = not flag_blank       
        event.clearEvents()
    
    
def cam_proj_align(ser, mon, screen_id):
    """Manually guide of the image to align the projector and the camera.
    The goal is to bring the fish into the center of view of the camera. 
    For helping purpose, it is projected a point in the center of the camera and a square that corresponds to the camera-ROI.

    Parameters
    ----------
    ser : serial comunication with Shapeoko
    mon : monitor object
    screen_id : 1

    Note
    ----------
    Press 's' to exit.

    """
    dl=1 
    
    win=visual.Window(size=mon.getSizePix(),monitor=mon,fullscr=True,screen=screen_id,units='cm',color='black')
    ref_pos=np.array([0, 0])
    circle=visual.Circle(win=win, radius=0.2, fillColor='green', pos=ref_pos)
    rect=visual.Rect(win=win, width=1.5, height=1.5, fillColor='white', pos=ref_pos)

    while True: 

        rect.draw()
        circle.draw()
        win.flip()  

        key=event.getKeys()
        if key==['s']:
            break

        if key==['up']:
            ShapeokoMove(ser,-dl,0)
        if key==['down']:
            ShapeokoMove(ser,dl,0)
        if key==['right']:
            ShapeokoMove(ser,0,dl)
        if key==['left']:
            ShapeokoMove(ser,0,-dl)

        event.clearEvents()

    win.close()