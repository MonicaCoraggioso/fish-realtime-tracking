"""
    Set of functions that allow to project a light pattern in the center of the camera, i.e. where the fish is located.
    
    Author: Monica Coraggioso
"""
import numpy as np
from psychopy import visual,event,clock,monitors,core
import serial, time #pySerial
import re #regex
from utilities.Shapeoko import ShapeokoMove, ShapeokoStatus, ShapeokoWrite

def manual_monitor_red(ser, win, screen_id, dl, x_center_motor, y_center_motor):
    """
    This function allows for the manual changing of the motor position using the keybord (arrows up-down-left-right). It also displays a red circle below.
    
    ser: serial port comunication with the motor
    mon: monitor id
    dl: step size
    x_center_now, y_center_now: centering motor-monitor
    """
    #win=visual.Window(size=mon.getSizePix(),monitor=mon,fullscr=True,screen=screen_id,units='cm',color='black')
    ref_pos=np.array([0, 0])
    circle=visual.Circle(win=win, radius=2, fillColor='red', pos=ref_pos)

    stop=False
    ser.write(b'G91\n')

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
 
        circle.draw()
        win.flip()  

        key=event.getKeys()
        if key==['s']:
            stop=True
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

        event.clearEvents()

    win.close()

    return stop