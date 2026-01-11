import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt

# lists for storing position data to later plot
x = []
y = []
z = []

# can use live frames from camera or frames from videos
# cap = cv.VideoCapture('C:/absolute/path/to/video.mp4')
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # may need to rotate or flip frame depending on video capture device
    # frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    # frame = cv.flip(frame, 1)

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV) # converts frame to HSV (Hue Saturation Value) format

    # lower bound for green, yellow (bright), and yellow (dim) HSV masks, in order
    # lower = (30, 0.7 * 255, 0.6 * 255)
    lower = (15, 0.7 * 255, 0.5 * 255)
    # lower = (15, 0.6 * 255, 0.3 * 255)

    # upper bound for green and yellow HSV masks, in order
    # upper = (55, 255, 255)
    upper = (35, 255, 255)

    mask = cv.inRange(hsv, lower, upper) # mask produces binary image to depict pixels that fall in between lower and upper bounds
    color = cv.bitwise_and(frame, frame, mask=mask) # shows the pixels that fall in between lower and upper bounds
    blur = cv.medianBlur(mask, 5) # blur to remove color noise that may affect results

    if cv.countNonZero(blur):
        nonzero = cv.findNonZero(blur) # nonzero pixels are pixels that fall in between lower and upper bounds
        cx, cy = np.astype(nonzero.mean(0)[0], 'int32') # the mean position of this pixels should approximate the center of the ball
        radius = np.round(np.sqrt(nonzero.shape[0] / np.pi)).astype('int32') # by treating pixel count as area, and since the ball is a circle, area = pi * radius ^ 2 => radius = sqrt(area / pi)
        scale = 0.037125 / radius # real radius of ball divided by pixel radius

        # Calculation for depth arises from pinhole camera optics and the triangle similarity: object height / object distance = image height / image distance.
        # Since image distance should be constant for a pinhole camera (and assumed constant for a webcamera), then solving for image distance => image distance = object distance * image height / object height.
        # Thus, image distance is a constant that can be calculated using three measureable values and can be used to solve for other object distances using the triangle similarity solved for object distance,
        # object distance = object height * image distance / image height. Note: scale = object height / image height.
        # In this code, depth is object distance, radius is image height, 0.037125 is object height, and 1080 (for my webcam, or 600 for my phone) are empirically calculated image distances.
        # For more information, check out https://pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/.
        
        depth = scale * 1080
        # depth = scale * 600
        
        cv.circle(frame, (cx, cy), radius, (0, 0, 255), 1) # draws a circle around where the ball should be

        cv.line(frame, (cx, 0), (cx, frame.shape[0]), (0, 0, 255), 1) # draws horizontal and vertical lines intersecting the ball's center
        cv.line(frame, (0, cy), (frame.shape[1], cy), (0, 0, 255), 1)

        # display pixel position, pixel radius, and approximate distance from camera on screen
        cv.putText(frame, f'({cx}, {cy}) {radius}px', (10, 30), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(frame, f'{np.round(depth, 2)}m', (10, 60), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2, cv.LINE_AA)

        # save position data for plotting
        x.append((cx - frame.shape[1] / 2) * scale)
        y.append((frame.shape[0] / 2 - cy) * scale)
        z.append(-depth)

    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('color', color)

    # press 'q' to close camera and windows
    if cv.waitKey(1) == ord('q'):
        break

    time.sleep(1 / 60)

cap.release()
cv.destroyAllWindows()

# creates a 3D plot of ball position, very susceptible to lighting
ax = plt.figure().add_subplot(projection='3d')
ax.plot(x, z, y)
ax.set_xlim(-10, 10)
ax.set_ylim(-20, 0)
ax.set_zlim(-10, 10)
plt.tight_layout()
plt.show()