import cv2
import numpy as np

img1= cv2.imread('../data/Undistort/87.jpg', cv2.IMREAD_GRAYSCALE)
img1_resize=cv2.resize(img1,(600,400))
img2= cv2.imread('../data/Undistort/131.jpg', cv2.IMREAD_GRAYSCALE)
img2_resize=cv2.resize(img2,(600,400))

orb= cv2.ORB_create() #tinh diem constant bang orb
key1,des1=orb.detectAndCompute(img1_resize,None)
key2,des2=orb.detectAndCompute(img2_resize,None)

#brute force matching
brute=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches=brute.match(des1, des2)
matches=sorted(matches,key= lambda x:x.distance) #tao order de lay diem khoang cach min

for m in matches:
    print (m.distance)

result=cv2.drawMatches(img1_resize,key1,img2_resize,key2,matches[:10], None,flags=2)

cv2.imshow("result",result)
cv2.waitKey(0)

