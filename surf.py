import cv2
import numpy as np
MIN_MATCH_COUNT=75

detector=cv2.SURF(3000)

FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

trainImg=cv2.imread("TrainingData/TrainImg.jpg",0)
trainKP,trainDesc=detector.detectAndCompute(trainImg,None)

cam=cv2.VideoCapture(0)
while True:
    ret, QueryImgBGR=cam.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
   
    matches=flann.knnMatch(queryDesc,trainDesc,k=2)
    print(len(matches))

    goodMatch=[]
    for m,n in matches:
        if m.distance<1*n.distance:
            goodMatch.append(m)
    if(len(goodMatch)>MIN_MATCH_COUNT):
        tp=[trainKP[m.trainIdx].pt for m in goodMatch]
        qp=[queryKP[m.queryIdx].pt for m in goodMatch]
        tp,qp=np.float32((tp,qp))
        H,mask=cv2.findHomography(tp,qp,cv2.RANSAC,5.0)
        matchesMask=mask.ravel().tolist()
        h,w=trainImg.shape
        trainBorder=np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),3)
    else:
        print "Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT)
        matchesmask=None
    cv2.imshow('result',QueryImgBGR)
    if cv2.waitKey(10)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
