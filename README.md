# DroneGestureControl
Controling a Tello Drone using a Hand Gesture Detector using a preprocessed model using OpenCV and VGG.

## Gestures
![gestures](https://miro.medium.com/max/1400/0*az-wcj3bJfqr50l6.jpg)

Gesture | Drone Function
------------ | -------------
Peace | Rotate 90° to the left
Palm | Move down 50cm
Okay | Rotate 90° to the right
L | Move up 50cm
Fist | Move forward 50cm

## Demo 
[![demo](https://img.youtube.com/vi/-GkevpPNNDg/0.jpg)](https://www.youtube.com/watch?v=-GkevpPNNDg)

## Instructions

For an optimal execution, try to be in a bright place with a flat background.

1. Connect computer to tello drone network
2. Run **gestures_drone.py**
3. Select the live video window
4. Leave as empty as you can the selected area for the gestures
5. Type **"b"** to capture the background and help the algorithm to detect only the hand
6. Make a gesture on the selected area and press **SPACE**
7. Repeat step 6 as you want
8. To reset the background, type **"r"** and then **"b"**
9. Type **ESC** to exit program and land drone

## Configuration

Python version: 3.7.6

Required packages:

```
  tensorflow==2.0.0  
  cv2(opencv)==3.4.7
  keras==2.3.1
  h5py==2.10.0
  numpy
  scipy
  pillow
  scikit-learn
```


