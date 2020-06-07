"""
    Author: Marco Antonio Mancha Alfaro
    
    Gesture hand detection to control a Tello Drone
    
    Credits to @athena15 for the VGG Model
"""
from keras.models import load_model
import cv2
import copy
import numpy as np
import time
import socket
import threading
import sys

##################################################
#              TELLO DRONE CODE                  #
##################################################

# IP and port of Tello
tello_address = ('192.168.10.1', 8889)

# IP and port of local computer
local_address = ('', 9000)

# Create a UDP connection that we 'll send the command to
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind to the local address and port
sock.bind(local_address)

# Send the message to Tello and allow for a delay in seconds
def send(message): 
    #Try to send the message otherwise print the exception
    try:
        sock.sendto(message.encode(), tello_address)
        print("Sending message: " + message)
    except Exception as e:
        print("Error sending: " + str(e))
        

# Receive the message from Tello
def receive(): 
    #Continuously loop and listen for incoming messages
    while True: 
        #Try to receive the message otherwise print the exception
        try:
            response, ip_address = sock.recvfrom(128)
            print("Received message: " + response.decode(encoding = 'utf-8'))
        except Exception as e: #If there 's an error close the socket and break out of the loop
            sock.close()
            print("Error receiving: " + str(e))
            break

# Function to send initial functions to drone
def takeoff():
    send("command")
    send("takeoff")

# Land Drone
def land():
    send("land")

# Create and start a listening thread that runs in the background# This utilizes our receive function and will continuously monitor for incoming messages
receiveThread = threading.Thread(target = receive)
receiveThread.daemon = True
receiveThread.start()

# Initializing drone variable
delay = 1
forward_distance = 50
angle = 90

##################################################
#           HAND GESTURE DETECTOR CODE           #
##################################################

model = load_model('model.h5')

gesture_names = {0: 'Fist',
                 1: 'L',
                 2: 'Okay',
                 3: 'Palm',
                 4: 'Peace'}

# initializing global variables 
cap_region_x_begin = 0.5  
cap_region_y_end = 0.8  
threshold = 60
blurValue = 41
bgSubThreshold = 50
isBgCaptured = 0  
triggerSwitch = False
action = ""
prediction = 0
score = 0

# Obtain a label prediction and probability from an image
def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    # Normalize
    image /= 255
    pred_array = model.predict(image)
    result = gesture_names[np.argmax(pred_array)]
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    return result, score


# Remove background from frame using erode and masking
def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=0)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

# Start video camera
cam_capture = cv2.VideoCapture(0)
takeoff()

while True:
    # Read frame
    ret, frame = cam_capture.read()
    # smoothing filter
    frame = cv2.bilateralFilter(frame, 5, 50, 100) 
     # flip the frame horizontally
    frame = cv2.flip(frame, 1) 
    # Show ROI
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

    # Show frame
    cv2.imshow('original', frame)

    # Run once background is captured
    if isBgCaptured == 1:
        img = remove_background(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
              int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Put text into window
        cv2.putText(thresh, f"Prediction: {prediction} ({score}%)", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255))
        cv2.putText(thresh, f"Action: {action}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255))  # Draw the text
        cv2.imshow('ori', thresh)

        # get the contours
        thresh1 = copy.deepcopy(thresh)
        _, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  
                # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i
            
            # Draw contours
            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

        cv2.imshow('output', drawing)

    # Keyboard OP
    k = cv2.waitKey(10)
    
    if k == 27:  
        # press ESC to exit all windows at any time
        land()
        sock.close()
        break
    
    elif k == ord('b'):  
        # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        time.sleep(2)
        isBgCaptured = 1
        print('Background captured')
        
    elif k == ord('r'): 
        # press 'r' to reset the background
        time.sleep(1)
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print('Reset background')
        
    elif k == 32:
        # press 'space' to capture drone move
        cv2.imshow('original', frame)
        # copies 1 channel BW image to all 3 RGB channels
        target = np.stack((thresh,) * 3, axis=-1)
        target = cv2.resize(target, (224, 224))
        # Reshape for VGG
        target = target.reshape(1, 224, 224, 3)
        # Obtain predictions
        prediction, score = predict_rgb_image_vgg(target)
        
        # Move tello drone based on the prediction
        if prediction == 'Fist':
            action = "forward " + str(forward_distance)
        elif prediction == 'Okay':
            action = "cw " + str(angle)
        elif prediction == 'Peace':
            action = "ccw " + str(angle)
        elif prediction == 'L':
            action = "up " + str(forward_distance)
        elif prediction == 'Palm':
            action = "down " + str(forward_distance)
        
        # Send detected move
        send(action)

# Close windows
cv2.destroyAllWindows()
cam_capture.release()
