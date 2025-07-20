print ("[YOLO] Security Surveillance Object Detection")
print ("[YOLO] Loading libraries...", flush=True)
import torch # refer to pytorch homepage for the current install commands (GPU dependent)
import cv2 # pip install opencv-python
import os
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO # pip install ultralytics

import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(2) # enable HIDPI support in Windows

MODEL = "yolo11n.pt"
CAMERA_URL = "rtsp://192.168.1.100:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream"
LOCATION = "FRONT" # Camaer LOCATION ID and filename
#SAVEPATH = r'/home/guest/Pictures/'
SAVEPATH = r'C:/Users/Downloads/Camera/'

if not os.path.exists(SAVEPATH):
    print(f"The save path '{SAVEPATH}' does not exist.")
    exit()

# Detection zone ROI(Region of interest) rectangle definition
ROI_x1=0.1 # ROI left bound
ROI_x2=0.7 # ROI right bound
ROI_y1=0.4 # ROI upper bound
ROI_y2=0.9 # ROI lower bound 

# Display constants for detection zone marker 
LINE_WIDTH=1 
RED_LINE_WIDTH=1
RED = (0,0,255) 
GREEN = (0,255,0)

# last_tile = np.array([])
tile4x4 = [] # event history tiles
savecount = 0 # event save count
display_toggle=1
key=0

# Test Rectangle Overlap [x1, y1, x2, y2]
class RECT:
    def isRectangleOverlap(self, R1, R2):
        if (R1[0]>=R2[2]) or (R1[2]<=R2[0]) or (R1[3]<=R2[1]) or (R1[1]>=R2[3]):
            return False
        else:
            return True
rect = RECT()

print ("[YOLO] Loading models...", flush=True)
model = YOLO(MODEL)
print (f"[YOLO] Model {MODEL} loaded.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda': print('[COMPUTE]', torch.cuda.get_device_name(0))
else: print(f'[COMPUTE] Using CPU.')

print ("[CAMERA] Connecting... Check camera IP address if failed.")
while True:
    cap = cv2.VideoCapture(CAMERA_URL)
    ret, img = cap.read() # print("ret",ret,img) # False None
    if ret:
        w = img.shape[1]
        h = img.shape[0]
        rx1=int(ROI_x1*w)
        ry1=int(ROI_y1*h)
        rx2=int(ROI_x2*w)
        ry2=int(ROI_y2*h)
        print ("[CAMERA] Connected.", img.shape)
        break
    else:
        print ("[CAMERA] Retrying...")
        time.sleep(5)

while True: # Main Loop
    ## start = time.time() # measure capture time
    now = datetime.now()
    ret = cap.grab() # Skip 1 frame
    # ret = cap.grab() # Skip 1 frame
    # ret = cap.grab() # Skip 1 frame
    ret, img = cap.read()
    if ret is not True: # Capture failed. Loop until camera recovers
        # print(type(ret)) 
        # print(ret) # ch, cw, cc = img.shape # True (1440, 2560, 3) #'NoneType' object has no attribute 'shape'
        print ("[CAMERA] Retrying...")
        time.sleep(5)
        continue
    
    ## start = time.time() # measure model compute time
    p = model(img, verbose=False)
    n = len(p[0].boxes) # number of detected items
    ## [DEBUG] print("Number of detected objects =", n)
    ## end = time.time()
    ## print ("Time(s)=",end-start)

    event = 0
    for i in range(n): # check bounding boxes of each newly detected objects
        x1=int(p[0].boxes.xyxy[i][0]) 
        y1=int(p[0].boxes.xyxy[i][1])
        x2=int(p[0].boxes.xyxy[i][2])
        y2=int(p[0].boxes.xyxy[i][3])
        img = cv2.rectangle(img,(x1,y1),(x2,y2), RED, LINE_WIDTH, cv2.LINE_AA) # draws red rectangles around ALL detected objects
        if rect.isRectangleOverlap([rx1, ry1, rx2, ry2],[x1,y1,x2,y2]): # ROI overlap detection
            # Known false positive classes, filter and skip
            if int(p[0].boxes.cls[i]) == 10: continue # fire hydrant 10
            if int(p[0].boxes.cls[i]) == 12: continue # parking meter 12
            if int(p[0].boxes.cls[i]) == 61: continue # toilet
            if int(p[0].boxes.cls[i]) == 58: continue # plant
            if int(p[0].boxes.cls[i]) == 72: continue # frigerator 72
            if int(p[0].boxes.cls[i]) == 75: continue # vase 75
            if int(p[0].boxes.cls[i]) == 29: continue # frisbee 29
            if int(p[0].boxes.cls[i]) == 13: continue # bench 13
            print(savecount, now, p[0].names[int(p[0].boxes.cls[i])], int(p[0].boxes.cls[i])) # print detected object classes for trouble shooting
            event = event + 1

    # Force save once as reference image and test output folder
    if savecount == 0:
        event = 1

    if event > 0:
        # Save image file with time stamps in filename
        milsec = str(int(now.microsecond/1000))
        if len(milsec) == 2:
            milsec = "0" + milsec
        if len(milsec) == 1:
            milsec = "00" + milsec
        filename = LOCATION+"_"+now.strftime("%Y_%m%d_%H%M%S_")+milsec+'_'+str(savecount)+'.jpg'
        cv2.imwrite(SAVEPATH+filename, img) #[DEBUG] print(SAVEPATH+filename)
        savecount = savecount + 1

        # Put time stamps on 4x4 tile images for display
        timestamp = str(savecount) + " " + now.strftime("%H:%M:%S")
        last_tile = cv2.resize(img, None, fx=0.25, fy= 0.25, interpolation= cv2.INTER_LINEAR)
        SCALE = 1
        cv2.putText(last_tile, str(timestamp), (15*SCALE, 45*SCALE), cv2.FONT_HERSHEY_SIMPLEX, SCALE, (0,0,0), 12, cv2.LINE_AA)
        cv2.putText(last_tile, str(timestamp), (15*SCALE, 45*SCALE), cv2.FONT_HERSHEY_SIMPLEX, SCALE, (255,255,255), 2, cv2.LINE_AA)
        tile4x4.append(last_tile)
        if(len(tile4x4)>16): tile4x4.pop(0)

        ch, cw, cc = img.shape        
        canvas = np.zeros((ch,cw,3), np.uint8)       
        h, w, c = last_tile.shape
        
        for i in range(len(tile4x4)):
            canvas[h*(int(i/4)%4):h*(int(i/4)%4)+h-1, w*(i%4):w*(i%4)+w-1]=tile4x4[i][0:h-1, 0:w-1]
        
        cv2.namedWindow("Event history", flags=cv2.WINDOW_GUI_NORMAL) # cv2.WINDOW_NORMAL
        cv2.imshow("Event history",canvas)

        img = cv2.rectangle(img,(rx1,ry1),(rx2,ry2), RED, RED_LINE_WIDTH, cv2.LINE_AA) # Live display
    else:
        img = cv2.rectangle(img,(rx1,ry1),(rx2,ry2), GREEN, RED_LINE_WIDTH, cv2.LINE_AA)  # Live display

    if(display_toggle):
        timestamp = str(savecount) + " " + now.strftime("%H:%M:%S")
        SCALE = 2
        cv2.putText(img, str(timestamp), (15*SCALE, 45*SCALE), cv2.FONT_HERSHEY_SIMPLEX, SCALE, (0,0,0), 12, cv2.LINE_AA)
        cv2.putText(img, str(timestamp), (15*SCALE, 45*SCALE), cv2.FONT_HERSHEY_SIMPLEX, SCALE, (255,255,255), 2, cv2.LINE_AA)

        cv2.namedWindow("Live", flags=cv2.WINDOW_GUI_NORMAL) # cv2.WINDOW_NORMAL
        cv2.imshow("Live",img)

    key = cv2.waitKey(10)
    # if key!= -1: print('key=', key)
    if key == 27: # 'ESC' key
        print('ESC')
        cv2.destroyAllWindows()
        cv2.waitKey(10)
        break
    elif key == 32: # 'space' key
        print("Main display window activated/deactivated.", key)
        display_toggle = not display_toggle
    elif display_toggle == 0:
        if cv2.getWindowProperty('Live', cv2.WND_PROP_VISIBLE):
            cv2.destroyWindow('Live')
    elif display_toggle == 1:
        if cv2.getWindowProperty('Live', cv2.WND_PROP_VISIBLE) < 1:
            print("Main display window closed manually.")
            display_toggle=0
