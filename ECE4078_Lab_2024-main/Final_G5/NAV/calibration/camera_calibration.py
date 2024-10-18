# for calculating the camera matrix from a photo of the calibration rig
import numpy as np
import os
import sys
import re
import matplotlib
import matplotlib.pyplot as plt

from machinevisiontoolbox import Image, CentralCamera

if __name__ == '__main__':
    
    # Display image
    img = Image.Read('./calib_0.png', grey=True)
    image = Image(img)
    fig = matplotlib.pyplot.figure()
    plt.imshow(image.image, cmap='gray')
    
    # Variables, p will contains clicked points, idx contains current point that is being selected
    p = np.ones((12,2)) * -1
    idx = 0

    
    # pick points
    def onclick(event):
        global p, idx
        
        if event.button == 1:
            # left mouse click, add point and increment by 1
            p[idx, 0] = event.xdata
            p[idx, 1] = event.ydata
            idx = idx + 1
        elif event.button == 3:
            # right click, go back to previous point
            idx -= 1
            p[idx, 0] = -1
            p[idx, 1] = -1
            
        idx = np.minimum(np.maximum(idx, 0), 11) # to keep within bounds
        print(str(p.T))
    
    print("Specify points on the calibration rig following order")
    fig.canvas.manager.set_window_title('Close image window once all 8 points are selected')    
    ka = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    p = p.T

    cm = 0.01 # centimetre to metre conversion factor
    
    P_calib = np.array([
        [ 0,  -13.5, 12.2],
        [ 0,   -7.5, 12.2],
        [ 0,  -13.5,  6.2],
        [ 0,   -7.5,  6.2],
        [ 6.8,  0,   12.75],
        [12.8,  0,   12.65],
        [ 6.55,  0,    6.8],
        [12.55,  0,    6.65],
        [6.55,  -6.4,   0],
        [12.55,  -6.6,    0],
        [6.35,  -12.35,    0],
        [12.3,  -12.6,    0]
    ]).T * cm # calibration rig specs
    
    # compute the camera matrix
    C, _ = CentralCamera.points2C(P_calib, p)
    camera = CentralCamera.decomposeC(C)
    
    print("\nCamera info:\n", camera)
    
    # save the intrinsic parameters 
    dataDir = "{}/param/".format(os.getcwd())
    print("\nIntrinsic parameters:\n", camera.K)
    fileNameI = "{}intrinsic.txt".format(dataDir)
    np.savetxt(fileNameI, camera.K, delimiter=',')
    
    # extrinsic parameters
    # print("\nExtrinsic parameters:\n", repr(camera.pose))

# The baseline parameter is estimated as 0.074357 m.
# The scale parameter is estimated as 0.002670 m/ticks.


# [[319.04854485   0.         160.49936155]
# [  0.         318.8325795  119.3245279 ]
# [  0.           0.           1.        ]]