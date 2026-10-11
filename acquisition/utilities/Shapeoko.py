"""
    Serial port comunication with Carbide 3D Shapeoko 4 using G-code.
    
    Author: Monica Coraggioso
"""
import os
import serial, time #pySerial
import numpy as np

def ShapeokoMove(ser, deltaX, deltaY, speed=None): 
    """Send to ser a command of movement.
    
    Parameters
    ----------
    ser : serial comunication with Shapeoko
    deltaX : step in x [mm]
    deltaY : step in y [mm]
    speed : desired speed [mm/min]. If speed=None maximum speed is used.
  
    """
    if speed==None:
        tmp = 'G00X' + str(deltaX) + 'Y' + str(deltaY) + '\n'
    else:
        tmp = 'G1X' + str(deltaX) + 'Y' + str(deltaY) + 'F' + str(speed) + '\n'
    ser.write(str.encode(tmp)) #.encode: str2bytes
    
def ShapeokoStatus(ser):
    """Real-time status report at 20Hz.
    
    Parameters
    ----------
    ser : serial comunication with Shapeoko
    
    Returns:
    --------
    msg : <str>. Important informations (see 'https://github.com/grbl/grbl/wiki/Interfacing-with-Grbl#streaming-a-g-code-program-to-grbl'):
        -WPos:0.000,0.000,0.000 : Work position listed as X,Y,Z coordinates [mm]
        -Buf:0 : Number of motions queued in Grbl's planner buffer.
    """
    ser.flushInput()
    ser.write(b'?')
    time.sleep(0.05) 
    tmp = ser.readline()
    msg = tmp.decode() #.decode: bytes2str
    return msg

def ShapeokoWrite(ser, message, print_msg=None): 
    """Real-time status report at 20Hz. Equivalent to 'ShapeokoStatus' if ShapeokoWrite(ser,'?')
    
    Parameters
    ----------
    ser : serial comunication with Shapeoko
    
    Returns
    --------
    msg : <str>
    """
    ser.flushInput()
    ser.write(str.encode(message+'\n'))
    time.sleep(0.05) #20Hz
    msg_status=[]
    while ser.in_waiting > 0:
        tmp = ser.readline()
        msg = tmp.decode()
        if print_msg:
            print(msg)
        if msg[0]=='<':
            msg_status=msg
    return msg_status
        
def ShapeokoStart(port=None): 
    """Inizialize the serial connection and set the default parameters for Shapeoko (see 'https://docs.carbide3d.com/software-faq/shapeoko-3-default-grbl-settings/')
    
    Returns
    ----------
    ser : serial comunication with Shapeoko
    """
    # Open the serial connection. Port defaults to $SHAPEOKO_PORT, else COM5 (Windows lab PC).
    port = port or os.environ.get("SHAPEOKO_PORT", "COM5")
    ser = serial.Serial(port, 115200, timeout=1)
    if not ser.isOpen():
        ser.open()
    print(f'{port} is open', ser.isOpen())

    #Set all the default parameters; see 'https://docs.carbide3d.com/software-faq/shapeoko-3-default-grbl-settings/'
    time.sleep(1)
    ser.write(b'$0=10\n')
    time.sleep(0.01)
    ser.write(b'$1=255\n')
    time.sleep(0.01)
    ser.write(b'$2=0\n')
    time.sleep(0.01)
    ser.write(b'$3=0\n')
    time.sleep(0.01)
    ser.write(b'$4=0\n')
    time.sleep(0.01)
    ser.write(b'$5=3\n')
    time.sleep(0.01)
    ser.write(b'$6=0\n')
    time.sleep(0.01)
    ser.write(b'$10=6\n') #status report mask
    time.sleep(0.01)
    ser.write(b'$11=0.020\n')
    time.sleep(0.01)
    ser.write(b'$12=0.010\n')
    time.sleep(0.01)
    ser.write(b'$13=0\n')
    time.sleep(0.01)
    ser.write(b'$20=1\n')
    time.sleep(0.01)
    ser.write(b'$21=0\n')
    time.sleep(0.01)
    ser.write(b'$22=9\n')
    time.sleep(0.01)
    ser.write(b'$23=3\n')
    time.sleep(0.01)
    ser.write(b'$24=100.000\n')
    time.sleep(0.01)
    ser.write(b'$25=3000.000\n')
    time.sleep(0.01)
    ser.write(b'$26=25\n')
    time.sleep(0.01)
    ser.write(b'$27=2.000\n')
    time.sleep(0.01)
    ser.write(b'$30=1000\n')
    time.sleep(0.01)
    ser.write(b'$31=0\n')
    time.sleep(0.01)
    ser.write(b'$32=0\n')
    time.sleep(0.01)
    ser.write(b'$43=1\n')
    time.sleep(0.01)
    ser.write(b'$44=3\n')
    time.sleep(0.01)
    ser.write(b'$100=640.000\n') #X-axis steps per millimeter
    time.sleep(0.01)
    ser.write(b'$101=640.000\n') #Y-axis steps per millimeter
    time.sleep(0.01)
    ser.write(b'$102=200.000\n')
    time.sleep(0.01)
    ser.write(b'$110=10000.000\n') #X-axis maximum rate, mm/min
    time.sleep(0.01)
    ser.write(b'$111=10000.000\n') #Y-axis maximum rate, mm/min
    time.sleep(0.01)
    ser.write(b'$112=1000.000\n')
    time.sleep(0.01)
    ser.write(b'$120=100.000\n') #X-axis acceleration, mm/min^2 (max.500.000). (usually 100.000)
    time.sleep(0.01)
    ser.write(b'$121=100.000\n') #Y-axis acceleration, mm/min^2 (max.500.000). (usually 100.000)
    time.sleep(0.01)
    ser.write(b'$122=270.000\n')
    time.sleep(0.01)
    ser.write(b'$130=400.000\n') #X-axis maximum travel, millimeters
    time.sleep(0.01)
    ser.write(b'$131=445.000\n') #Y-axis maximum travel, millimeters
    time.sleep(0.01)
    ser.write(b'$132=95.000\n')
    
    return ser


def ShapeokoHoming(ser, reset_origin=None):   
    """Homing cycle and set homing position = origin of the axes coordinates
    
    Parameters
    ----------
    ser : serial comunication with Shapeoko
    """
    ser.write(b'$H\n')  
    while True:
        msg=ShapeokoStatus(ser)
        if msg:  
            print('Homing is over')
            if reset_origin!=None:
                time.sleep(0.1)
                ser.write(b'g92x0y0\n') # reset the axes origin
                print('Resetting the origin: now A=(0,0)')
            print(ShapeokoStatus(ser))
            break
            
def ShapeokoKill(ser):  
    """Kill the serial connection with Shapeoko
    
    Parameters
    ----------
    ser : serial comunication with Shapeoko
    """
    ser.close()
    
def ShapeokoBoundary(x_mot_now, y_mot_now, mot_xA, mot_yA, mot_xB, mot_yB, mot_xC, mot_yC, mot_xD, mot_yD, step_x, step_y):   
    """Define fisical boundary for the motor, they correspond to the inner agar frame. 
    The function take the current position of the motor and decide if a step=(step_x,step_y) can be performed by checking the boundary. 
    
    Parameters
    ----------
    x_mot_now, y_mot_now = current motor position
    mot_xA, mot_yA = close to the homing
    mot_xB, mot_yB = close to maximum x
    mot_xC, mot_yC = close to maximum x and maximum y
    mot_xD, mot_yD = close to maximum y
    step_x, step_y = step
    
    Returns:
    --------- 
    flag_inside = True (/False), motor inside (/outside) the border->the step is (not) allowed
     
    """
    
    if mot_xA <= (x_mot_now + step_x) and mot_xB >= (x_mot_now + step_x) and mot_yA <= (y_mot_now + step_y) and mot_yD >= (y_mot_now + step_y):
        flag_inside=True
    else:
        flag_inside=False
        
    return flag_inside
            
    
def ShapeokoBoundary2(ser, x_mot_now, y_mot_now, mot_xA, mot_yA, mot_xB, mot_yD, step_x, step_y, speed=None, dt_motor=None, r=None):
    """Define fisical boundary for the motor, they correspond to the inner agar frame. 
    The function take the current position of the motor and decide if a step=(step_x,step_y) can be performed by checking the boundary. 
    
    Parameters:
    ----------
    ser : serial comunication with Shapeoko
    x_mot_now, y_mot_now = current motor position
    mot_xA, mot_yA = close to the homing
    mot_xB, mot_yB = close to maximum x
    mot_xC, mot_yC = close to maximum x and maximum y
    mot_xD, mot_yD = close to maximum y
    step_x, step_y = step
    speed=None, dt_motor=None, r=None -> only for the harmonic tracking algorithm
    
    Inside:
    --------- 
    ShapeokoMove(ser,DeltaX,DeltaY,speed) if x_mot_now+step_x, y_mot_now+step_y are inside the boundaries
    ShapeokoMove(ser,'only in the correct direction') close to the boundaries. [N.B. Absolute mode!]
    ShapeokoMove=off close to the edges
     
    """
    if speed==None:
        DeltaX = x_mot_now+step_x
        DeltaY = y_mot_now+step_y
    else:
        speed = round(speed,0)
        if speed>0:
            step_x_osc = step_x/r*speed*dt_motor
            step_y_osc = step_y/r*speed*dt_motor
            DeltaX = x_mot_now+step_x_osc
            DeltaY = y_mot_now+step_y_osc
        else:
            DeltaX = x_mot_now
            DeltaY = y_mot_now
            speed=None
    DeltaX = round(DeltaX,3)
    DeltaY = round(DeltaY,3)
    
    
    if mot_xA <= (x_mot_now + step_x) and mot_xB >= (x_mot_now + step_x) and mot_yA <= (y_mot_now + step_y) and mot_yD >= (y_mot_now + step_y):
        #print('in')
        if speed==None:
            ShapeokoMove(ser,DeltaX,DeltaY)
        else:
            ShapeokoMove(ser,DeltaX,DeltaY,speed)
            
    elif mot_xA > (x_mot_now + step_x) and mot_xB >= (x_mot_now + step_x) and mot_yA <= (y_mot_now + step_y) and mot_yD >= (y_mot_now + step_y):
        #print('border1',DeltaY)
            if speed==None:
                if step_y>0.5:
                    ShapeokoMove(ser,x_mot_now,DeltaY)
            else:
                if step_y_osc>0.5:
                    ShapeokoMove(ser,x_mot_now,DeltaY,speed)
        
    elif mot_xA <= (x_mot_now + step_x) and mot_xB < (x_mot_now + step_x) and mot_yA <= (y_mot_now + step_y) and mot_yD >= (y_mot_now + step_y):
        #print('border2',DeltaY)
            if speed==None:
                if step_y>0.5:
                    ShapeokoMove(ser,x_mot_now,DeltaY)
            else:
                if step_y_osc>0.5:
                    ShapeokoMove(ser,x_mot_now,DeltaY,speed)
        
    elif mot_xA <= (x_mot_now + step_x) and mot_xB >= (x_mot_now + step_x) and mot_yA > (y_mot_now + step_y) and mot_yD >= (y_mot_now + step_y):
        #print('border3', DeltaX )
        if speed==None:
            if step_x>0.5:
                ShapeokoMove(ser,DeltaX,y_mot_now)
        else:
            if step_x_osc>0.5:
                ShapeokoMove(ser,DeltaX,y_mot_now,speed)
        
    elif mot_xA <= (x_mot_now + step_x) and mot_xB >= (x_mot_now + step_x) and mot_yA <= (y_mot_now + step_y) and mot_yD < (y_mot_now + step_y):
        #print('border4', DeltaX )
        if speed==None:
            if step_x>0.5:
                ShapeokoMove(ser,DeltaX,y_mot_now)
        else:
            if step_x_osc>0.5:
                ShapeokoMove(ser,DeltaX,y_mot_now,speed)
    else:
        print('edge, not move')
        
        
        
def ShapeokoFollow_simple(ser, x_mot_now, y_mot_now, mot_xA, mot_yA, mot_xB, mot_yD, step_x, step_y, speed=None):
    """Define fisical boundary for the motor, they correspond to the inner agar frame. 
    The function take the current position of the motor and decide if a step=(step_x,step_y) can be performed by checking the boundary. 
    
    Parameters:
    ----------
    ser : serial comunication with Shapeoko
    x_mot_now, y_mot_now = current motor position
    mot_xA, mot_yA = close to the homing
    mot_xB, mot_yB = close to maximum x
    mot_xC, mot_yC = close to maximum x and maximum y
    mot_xD, mot_yD = close to maximum y
    step_x, step_y = step
    speed=None, dt_motor=None, r=None -> only for the harmonic tracking algorithm
    
    Inside:
    --------- 
    ShapeokoMove(ser,DeltaX,DeltaY,speed) if x_mot_now+step_x, y_mot_now+step_y are inside the boundaries
    ShapeokoMove(ser,'only in the correct direction') close to the boundaries. [N.B. Absolute mode!]
    ShapeokoMove=off close to the edges
     
    """
    DeltaX = x_mot_now+step_x
    DeltaY = y_mot_now+step_y
           
    if mot_xA <= (x_mot_now + step_x) and mot_xB >= (x_mot_now + step_x) and mot_yA <= (y_mot_now + step_y) and mot_yD >= (y_mot_now + step_y):
        #print('in')
        if speed==None:
            ShapeokoMove(ser,DeltaX,DeltaY)
        else:
            ShapeokoMove(ser,DeltaX,DeltaY,speed)
            
    elif mot_xA > (x_mot_now + step_x) and mot_xB >= (x_mot_now + step_x) and mot_yA <= (y_mot_now + step_y) and mot_yD >= (y_mot_now + step_y):
        #print('border1',DeltaY)
            if speed==None:
                ShapeokoMove(ser,x_mot_now,DeltaY)
            else:
                ShapeokoMove(ser,x_mot_now,DeltaY,speed)
        
    elif mot_xA <= (x_mot_now + step_x) and mot_xB < (x_mot_now + step_x) and mot_yA <= (y_mot_now + step_y) and mot_yD >= (y_mot_now + step_y):
        #print('border2',DeltaY)
            if speed==None:
                ShapeokoMove(ser,x_mot_now,DeltaY)
            else:
                ShapeokoMove(ser,x_mot_now,DeltaY,speed)
        
    elif mot_xA <= (x_mot_now + step_x) and mot_xB >= (x_mot_now + step_x) and mot_yA > (y_mot_now + step_y) and mot_yD >= (y_mot_now + step_y):
        #print('border3', DeltaX )
        if speed==None:
            ShapeokoMove(ser,DeltaX,y_mot_now)
        else:
            ShapeokoMove(ser,DeltaX,y_mot_now,speed)
        
    elif mot_xA <= (x_mot_now + step_x) and mot_xB >= (x_mot_now + step_x) and mot_yA <= (y_mot_now + step_y) and mot_yD < (y_mot_now + step_y):
        #print('border4', DeltaX )
        if speed==None:
            ShapeokoMove(ser,DeltaX,y_mot_now)
        else:
            ShapeokoMove(ser,DeltaX,y_mot_now,speed)
    else:
        print('edge, not move')
        
        
def ShapeokoFollow_osc(ser, x_mot_now, y_mot_now, mot_xA, mot_yA, mot_xB, mot_yD, step_x, step_y, speed=None, dt_motor=None, r=None):
    """Define fisical boundary for the motor, they correspond to the inner agar frame. 
    The function take the current position of the motor and decide if a step=(step_x,step_y) can be performed by checking the boundary. 
    
    Parameters:
    ----------
    ser : serial comunication with Shapeoko
    x_mot_now, y_mot_now = current motor position
    mot_xA, mot_yA = close to the homing
    mot_xB, mot_yB = close to maximum x
    mot_xC, mot_yC = close to maximum x and maximum y
    mot_xD, mot_yD = close to maximum y
    step_x, step_y = step
    speed=None, dt_motor=None, r=None -> only for the harmonic tracking algorithm
    
    Inside:
    --------- 
    ShapeokoMove(ser,DeltaX,DeltaY,speed) if x_mot_now+step_x, y_mot_now+step_y are inside the boundaries
    ShapeokoMove(ser,'only in the correct direction') close to the boundaries. [N.B. Absolute mode!]
    ShapeokoMove=off close to the edges
     
    """
    if speed==None:
        DeltaX = x_mot_now+step_x
        DeltaY = y_mot_now+step_y
    else:
        speed = round(speed,0)
        if speed>0:
            step_x_osc = step_x/r*speed*dt_motor
            step_y_osc = step_y/r*speed*dt_motor
            DeltaX = x_mot_now+step_x_osc
            DeltaY = y_mot_now+step_y_osc
        else:
            DeltaX = x_mot_now
            DeltaY = y_mot_now
            speed=None
    DeltaX = round(DeltaX,3)
    DeltaY = round(DeltaY,3)
        
    if mot_xA <= (x_mot_now + step_x) and mot_xB >= (x_mot_now + step_x) and mot_yA <= (y_mot_now + step_y) and mot_yD >= (y_mot_now + step_y):
        #print('in')
        if speed==None:
            ShapeokoMove(ser,DeltaX,DeltaY)
        else:
            ShapeokoMove(ser,DeltaX,DeltaY,speed)
            
    elif mot_xA > (x_mot_now + step_x) and mot_xB >= (x_mot_now + step_x) and mot_yA <= (y_mot_now + step_y) and mot_yD >= (y_mot_now + step_y):
        #print('AD',DeltaY)
            if speed==None:
                if step_y>0.5:
                    ShapeokoMove(ser,mot_xA,DeltaY)
            else:
                if step_y_osc>0.5:
                    ShapeokoMove(ser,mot_xA,DeltaY,speed)
        
    elif mot_xA <= (x_mot_now + step_x) and mot_xB < (x_mot_now + step_x) and mot_yA <= (y_mot_now + step_y) and mot_yD >= (y_mot_now + step_y):
        #print('BC',DeltaY)
            if speed==None:
                if step_y>0.5:
                    ShapeokoMove(ser,mot_xB,DeltaY)
            else:
                if step_y_osc>0.5:
                    ShapeokoMove(ser,mot_xB,DeltaY,speed)
        
    elif mot_xA <= (x_mot_now + step_x) and mot_xB >= (x_mot_now + step_x) and mot_yA > (y_mot_now + step_y) and mot_yD >= (y_mot_now + step_y):
        #print('AB', DeltaX )
        if speed==None:
            if step_x>0.5:
                ShapeokoMove(ser,DeltaX,mot_yA)
        else:
            if step_x_osc>0.5:
                ShapeokoMove(ser,DeltaX,mot_yA,speed)
        
    elif mot_xA <= (x_mot_now + step_x) and mot_xB >= (x_mot_now + step_x) and mot_yA <= (y_mot_now + step_y) and mot_yD < (y_mot_now + step_y):
        #print('DC', DeltaX )
        if speed==None:
            if step_x>0.5:
                ShapeokoMove(ser,DeltaX,mot_yD)
        else:
            if step_x_osc>0.5:
                ShapeokoMove(ser,DeltaX,mot_yD,speed)
                
    elif (x_mot_now + step_x) <= mot_xA  and (y_mot_now + step_y) <= mot_yA:
        #print('A')
        if speed==None:
            if step_x>0.5:
                ShapeokoMove(ser,mot_xA,mot_yA)
        else:
            if step_x_osc>0.5:
                ShapeokoMove(ser,mot_xA,mot_yA,speed)
                
    elif (x_mot_now + step_x) >= mot_xB and (y_mot_now + step_y) <= mot_yB:
        #print('B')
        if speed==None:
            if step_x>0.5:
                ShapeokoMove(ser,mot_xB,mot_yB)
        else:
            if step_x_osc>0.5:
                ShapeokoMove(ser,mot_xB,mot_yB,speed)
                
    elif (x_mot_now + step_x) >= mot_xC and (y_mot_now + step_y) >= mot_yC:
        #print('C')
        if speed==None:
            if step_x>0.5:
                ShapeokoMove(ser,mot_xC,mot_yC)
        else:
            if step_x_osc>0.5:
                ShapeokoMove(ser,mot_xC,mot_yC,speed)
                
    elif (x_mot_now + step_x) <= mot_xD and (y_mot_now + step_y) >= mot_yD:
        #print('D')
        if speed==None:
            if step_x>0.5:
                ShapeokoMove(ser,mot_xD,mot_yD)
        else:
            if step_x_osc>0.5:
                ShapeokoMove(ser,mot_xD,mot_yD,speed)
        
        print('edge, not move')
        
def ShapeokoFollow_osc2(ser, x_mot_now, y_mot_now, mot_xA, mot_yA, mot_xB, mot_yD, step_x, step_y, speed=None, dt_motor=None, r=None):
    """Define fisical boundary for the motor, they correspond to the inner agar frame. 
    The function take the current position of the motor and decide if a step=(step_x,step_y) can be performed by checking the boundary. 
    
    Parameters:
    ----------
    ser : serial comunication with Shapeoko
    x_mot_now, y_mot_now = current motor position
    mot_xA, mot_yA = close to the homing
    mot_xB, mot_yB = close to maximum x
    mot_xC, mot_yC = close to maximum x and maximum y
    mot_xD, mot_yD = close to maximum y
    step_x, step_y = step
    speed=None, dt_motor=None, r=None -> only for the harmonic tracking algorithm
    
    Inside:
    --------- 
    ShapeokoMove(ser,DeltaX,DeltaY,speed) if x_mot_now+step_x, y_mot_now+step_y are inside the boundaries
    ShapeokoMove(ser,'only in the correct direction') close to the boundaries. [N.B. Absolute mode!]
    ShapeokoMove=off close to the edges
     
    """
    
    DeltaX = x_mot_now+step_x
    DeltaY = y_mot_now+step_y
        
    if mot_xA <= (x_mot_now + step_x) and mot_xB >= (x_mot_now + step_x) and mot_yA <= (y_mot_now + step_y) and mot_yD >= (y_mot_now + step_y):
        #print('in')
        if speed==None:
            ShapeokoMove(ser,DeltaX,DeltaY)
        else:
            ShapeokoMove(ser,DeltaX,DeltaY,speed)
        
        

def ShapeokoFollow_osc3(ser, x_mot_now, y_mot_now, mot_xA, mot_yA, mot_xB, mot_yD, step_x, step_y, speed=None):
    """Define fisical boundary for the motor, they correspond to the inner agar frame. 
    The function take the current position of the motor and decide if a step=(step_x,step_y) can be performed by checking the boundary. 
    
    Parameters:
    ----------
    ser : serial comunication with Shapeoko
    x_mot_now, y_mot_now = current motor position
    mot_xA, mot_yA = close to the homing
    mot_xB, mot_yB = close to maximum x
    mot_xC, mot_yC = close to maximum x and maximum y
    mot_xD, mot_yD = close to maximum y
    step_x, step_y = step
    speed=None, dt_motor=None, r=None -> only for the harmonic tracking algorithm
    
    Inside:
    --------- 
    ShapeokoMove(ser,DeltaX,DeltaY,speed) if x_mot_now+step_x, y_mot_now+step_y are inside the boundaries
    ShapeokoMove(ser,'only in the correct direction') close to the boundaries. [N.B. Absolute mode!]
    ShapeokoMove=off close to the edges
     
    """
    
    DeltaX = x_mot_now+step_x
    DeltaY = y_mot_now+step_y
        
    if mot_xA <= (x_mot_now + step_x) and mot_xB >= (x_mot_now + step_x) and mot_yA <= (y_mot_now + step_y) and mot_yD >= (y_mot_now + step_y):
        #print('in')
        if speed==None:
            ShapeokoMove(ser,DeltaX,DeltaY)
        else:
            ShapeokoMove(ser,DeltaX,DeltaY,speed)
            
    elif mot_xA > (x_mot_now + step_x) and mot_xB >= (x_mot_now + step_x) and mot_yA <= (y_mot_now + step_y) and mot_yD >= (y_mot_now + step_y):
        #print('AD',DeltaY)
        ShapeokoMove(ser,mot_xA,DeltaY,speed)
        
    elif mot_xA <= (x_mot_now + step_x) and mot_xB < (x_mot_now + step_x) and mot_yA <= (y_mot_now + step_y) and mot_yD >= (y_mot_now + step_y):
        #print('BC',DeltaY)
        ShapeokoMove(ser,mot_xB,DeltaY,speed)
        
    elif mot_xA <= (x_mot_now + step_x) and mot_xB >= (x_mot_now + step_x) and mot_yA > (y_mot_now + step_y) and mot_yD >= (y_mot_now + step_y):
        #print('AB', DeltaX )
        ShapeokoMove(ser,DeltaX,mot_yA,speed)
        
    elif mot_xA <= (x_mot_now + step_x) and mot_xB >= (x_mot_now + step_x) and mot_yA <= (y_mot_now + step_y) and mot_yD < (y_mot_now + step_y):
        #print('DC', DeltaX )
        ShapeokoMove(ser,DeltaX,mot_yD,speed)
                
    elif (x_mot_now + step_x) <= mot_xA  and (y_mot_now + step_y) <= mot_yA:
        #print('A')
        ShapeokoMove(ser,mot_xA,mot_yA,speed)
                
    elif (x_mot_now + step_x) >= mot_xB and (y_mot_now + step_y) <= mot_yB:
        #print('B')
        ShapeokoMove(ser,mot_xB,mot_yB,speed)
                
    elif (x_mot_now + step_x) >= mot_xC and (y_mot_now + step_y) >= mot_yC:
        #print('C')
        ShapeokoMove(ser,mot_xC,mot_yC,speed)
                
    elif (x_mot_now + step_x) <= mot_xD and (y_mot_now + step_y) >= mot_yD:
        #print('D')
        ShapeokoMove(ser,mot_xD,mot_yD,speed)
    

        
def ShapeokoFollow_tabs(ser, x_mot_now, y_mot_now, tab_x, tab_y, step_x, step_y, speed=None):
    """The function take the current position of the motor and decide if a step=(step_x,step_y) 
    can be performed by checking the boundary via tab_x and tab_y. 
    
    Parameters:
    ----------
    ser : serial comunication with Shapeoko
    x_mot_now, y_mot_now = current motor position (mm)
    tab_x = for each x_pos_motor you have the min_y_value and the max_y_value allowed->tab_x[int(mot_x),min_y,max_y]
    tab_y = for each y_pos_motor you have the min_x_value and the max_x_value allowed->tab_y[int(mot_y),min_x,max_x]
    step_x, step_y = step 
    speed = required speed evaluated with the harmonic equation. If None, motor moves at maximum speed.
    
    Inside:
    --------- 
    ShapeokoMove(ser,DeltaX,DeltaY,speed) if x_mot_now+step_x, y_mot_now+step_y are inside the boundaries
    ShapeokoMove(ser,'only in the correct direction') close to the boundaries. [N.B. Absolute mode!]

    Note:
    --------- 
    e.g. tab_x[0,.,.] is related to the minimum_motor_x_value(==mot_xA) which is NOT 0, that's why id_x0 is needed (same for y)
    """
    edge=False
    DeltaX = x_mot_now+step_x
    DeltaY = y_mot_now+step_y
    #print(DeltaX,DeltaY)
    x0=round(DeltaX)
    try:
        id_x0=np.where(tab_x[:,0]==x0)[0][0]
    except:
        id_x0=np.where(tab_x[:,0]==tab_x[-1,0])[0][0]

    y0=round(DeltaY)
    try:
        id_y0=np.where(tab_y[:,0]==y0)[0][0]
    except:
        id_y0=np.where(tab_y[:,0]==tab_y[-1,0])[0][0]
    #print(x0,y0)
    #print(id_x0,id_y0)
 
    if y0<tab_x[id_x0][1]:
        y0=tab_x[id_x0][1]
        edge=True
        #print(1,x0,y0)

    if y0>tab_x[id_x0][2]:
        y0=tab_x[id_x0][2] 
        edge=True
        #print(2,x0,y0)

    if x0<tab_y[id_y0][1]:
        x0=tab_y[id_y0][1]
        edge=True
        #print(3,x0,y0)

    if x0>tab_y[id_y0][2]:
        x0=tab_y[id_y0][2]
        edge=True
        #print(4,x0,y0)

    if edge:
        DeltaX,DeltaY = x0,y0
        
    if speed==None:
        ShapeokoMove(ser,DeltaX,DeltaY)
    else:
        ShapeokoMove(ser,DeltaX,DeltaY,speed)
    if edge:
        time.sleep(0.01)
    return edge
        
        
        
def ShapeokoFollow_TEST(ser, x_mot_now, y_mot_now, mot_xA, mot_yA, mot_xB, mot_yD, tab_x, tab_y, step_x, step_y, speed=None):
    """The function take the current position of the motor and decide if a step=(step_x,step_y) 
    can be performed by checking the boundary via tab_x and tab_y. 
    
    Parameters:
    ----------
    ser : serial comunication with Shapeoko
    x_mot_now, y_mot_now = current motor position (mm)
    tab_x = for each x_pos_motor you have the min_y_value and the max_y_value allowed->tab_x[int(mot_x),min_y,max_y]
    tab_y = for each y_pos_motor you have the min_x_value and the max_x_value allowed->tab_y[int(mot_y),min_x,max_x]
    step_x, step_y = step 
    speed = required speed evaluated with the harmonic equation. If None, motor moves at maximum speed.
    
    Inside:
    --------- 
    ShapeokoMove(ser,DeltaX,DeltaY,speed) if x_mot_now+step_x, y_mot_now+step_y are inside the boundaries
    ShapeokoMove(ser,'only in the correct direction') close to the boundaries. [N.B. Absolute mode!]

    Note:
    --------- 
    e.g. tab_x[0,.,.] is related to the minimum_motor_x_value(==mot_xA) which is NOT 0, that's why id_x0 is needed (same for y)
    """
    edge=False
    DeltaX = x_mot_now+step_x
    DeltaY = y_mot_now+step_y
    #print(DeltaX,DeltaY)
    x0=round(DeltaX)
    try:
        id_x0=np.where(tab_x[:,0]==x0)[0][0]
    except:
        id_x0=np.where(tab_x[:,0]==tab_x[-1,0])[0][0]

    y0=round(DeltaY)
    try:
        id_y0=np.where(tab_y[:,0]==y0)[0][0]
    except:
        id_y0=np.where(tab_y[:,0]==tab_y[-1,0])[0][0]
    #print(x0,y0)
    #print(id_x0,id_y0)

    if y0<tab_x[id_x0][1]:# or bAB:
        y0=mot_yA
        edge=True

    if y0>tab_x[id_x0][2]:# or bCD:
        y0=mot_yD
        edge=True

    if x0<tab_y[id_y0][1]:# or bDA:
        x0=mot_xA
        edge=True

    if x0>tab_y[id_y0][2]:# or bBC:
        x0=mot_xB
        edge=True
    
    if edge:
        print('EDGE',edge)
        DeltaX,DeltaY = x0,y0
        step_x,step_y=DeltaX-x_mot_now,DeltaY-y_mot_now
    if speed==None:
        ShapeokoMove(ser,DeltaX,DeltaY)
    else:
        ShapeokoMove(ser,DeltaX,DeltaY,speed)
    #if edge:
    #    time.sleep(0.1)
    return edge



def ShapeokoFollow_TEST2(ser, x_mot_now, y_mot_now, mot_xA, mot_yA, mot_xB, mot_yD, tab_x, tab_y, step_x, step_y, speed=None):
    """The function take the current position of the motor and decide if a step=(step_x,step_y) 
    can be performed by checking the boundary via tab_x and tab_y. 
    
    Parameters:
    ----------
    ser : serial comunication with Shapeoko
    x_mot_now, y_mot_now = current motor position (mm)
    tab_x = for each x_pos_motor you have the min_y_value and the max_y_value allowed->tab_x[int(mot_x),min_y,max_y]
    tab_y = for each y_pos_motor you have the min_x_value and the max_x_value allowed->tab_y[int(mot_y),min_x,max_x]
    step_x, step_y = step 
    speed = required speed evaluated with the harmonic equation. If None, motor moves at maximum speed.
    
    Inside:
    --------- 
    ShapeokoMove(ser,DeltaX,DeltaY,speed) if x_mot_now+step_x, y_mot_now+step_y are inside the boundaries
    ShapeokoMove(ser,'only in the correct direction') close to the boundaries. [N.B. Absolute mode!]

    Note:
    --------- 
    e.g. tab_x[0,.,.] is related to the minimum_motor_x_value(==mot_xA) which is NOT 0, that's why id_x0 is needed (same for y)
    """
    edge=False
    DeltaX = x_mot_now+step_x
    DeltaY = y_mot_now+step_y
    #print(DeltaX,DeltaY)
    x0=round(DeltaX)
    try:
        id_x0=np.where(tab_x[:,0]==x0)[0][0]
    except:
        id_x0=np.where(tab_x[:,0]==tab_x[-1,0])[0][0]

    y0=round(DeltaY)
    try:
        id_y0=np.where(tab_y[:,0]==y0)[0][0]
    except:
        id_y0=np.where(tab_y[:,0]==tab_y[-1,0])[0][0]
    #print(x0,y0)
    #print(id_x0,id_y0)

    if y0<tab_x[id_x0][1]:# or bAB:
        edge=True
        if y0<mot_yA:
            y0=mot_yA

    if y0>tab_x[id_x0][2]:# or bCD:
        edge=True
        if y0>mot_yD:
            y0=mot_yD

    if x0<tab_y[id_y0][1]:# or bDA:
        edge=True
        if x0<mot_xA:
            x0=mot_xA

    if x0>tab_y[id_y0][2]:# or bBC:
        edge=True
        if x0>mot_xB:
            x0=mot_xB
       
    if edge:
        #print('EDGE',edge)
        DeltaX,DeltaY = x0,y0
        step_x,step_y=DeltaX-x_mot_now,DeltaY-y_mot_now

    r=round( np.sqrt(step_x**2+step_y**2), 2)
    if r>0.1:                   
        if speed==None:
            ShapeokoMove(ser,DeltaX,DeltaY)
        else:
            ShapeokoMove(ser,DeltaX,DeltaY,speed)
    #if edge:
    #    time.sleep(0.1)
    return edge
        
      