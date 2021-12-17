from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import imutils
import os
import time

def boundary(img_1,img_2, diff_temp):
    thr = cv2.threshold(diff_temp, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        (x, y, h, w) = cv2.boundingRect(c)
        cv2.rectangle(img_1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img_2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("card 1", img_1)
    cv2.imshow("card 2", img_2)
    cv2.imshow("thresh", thr)
    cv2.waitKey(0)

#image_1_src=cv2.imread("data/Compare/beyond_card_ssim_1.jpg")
#image_2_src=cv2.imread("data/Compare/beyond_master_card_ssim.jpg")
#image_1=cv2.cvtColor(image_1_src,cv2.COLOR_BGR2GRAY)
#image_2=cv2.cvtColor(image_2_src,cv2.COLOR_BGR2GRAY)
#(score,diff)=ssim(image_1,image_2,full=True) #find similaries score between to pic
#diff=(diff*255).astype("uint8")
#boundary(image_1_src,image_2_src,diff)
#print(score,diff)

needle=cv2.imread("Test_lmTst_89.jpg",cv2.IMREAD_GRAYSCALE)

ssim_score=[]
start=time.time()
for i in range(192):
    img_compare=cv2.imread("D:/Data/lmTst_%d.jpg" %i, cv2.IMREAD_GRAYSCALE)
    score=ssim(needle,img_compare)
    print("lmTst_%d:" %i, score)
    ssim_score.append(score)
end=time.time()
print("calculation time:", end-start)
match=max(ssim_score)
index=ssim_score.index(match)
print("best match:", match, "location:", index)

#image_match=cv2.imread("D:/Data/lmTst_%d.jpg" %index)
#cv2.imshow("best match", image_match)
#cv2.waitKey(0)








