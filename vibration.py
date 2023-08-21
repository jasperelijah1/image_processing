import cv2
import numpy as np
import math

surf=cv2.SURF(500)
img=cv2.imread("thumb-1920-463479.png")
imgrs=cv2.resize(img,(400,300))
kp=surf.detect(img,None)
imgrs=cv2.drawKeypoints(imgrs,kp,None,(255,0,0),2)
pts = [k.pt for k in kp]
a=len(pts)
print ('The list of detected keypoints is:')
print a
for i in range(a):
    print "point",i,":", pts[i]
    y=pts[i][1]
    m=y-270.8289489746094
    if(m==0):
        print "Coincided points is",i,":",pts[i]
        x = pts[i][0]
        y = pts[i][1]
        a=int(x)
        b=int(y)
        
        cv2.circle(imgrs,(a,b),5,(0,0,255),-1)
        
        
    else:
        print "No points coincided"

cv2.line(imgrs,(0,180),(400,180),(0,255,0),3)
cv2.line(imgrs,(0,270),(400,270),(0,255,0),3)
cv2.imshow('res',imgrs)

cv2.waitKey(0)
