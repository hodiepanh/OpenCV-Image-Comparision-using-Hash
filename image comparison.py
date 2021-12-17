import cv2
import numpy as np
#histogram matching

image_src_base=cv2.imread("../data/Compare/Histogram_Comparison_Source_0.jpg")
image_src_1=cv2.imread("../data/Compare/Histogram_Comparison_Source_1.jpg")
image_src_2=cv2.imread("../data/Compare/Histogram_Comparison_Source_2.jpg")

base=cv2.cvtColor(image_src_base,cv2.COLOR_BGR2HSV)
image_1=cv2.cvtColor(image_src_1,cv2.COLOR_BGR2HSV)
image_2=cv2.cvtColor(image_src_2,cv2.COLOR_BGR2HSV)

base_half=base[base.shape[0]//2:,:]

#h_bins=50
#s_bins=60
hist_size=[50,60]

h_range=[0,180]
s_range=[0,256]
ranges= h_range+s_range

channel=[0,1]

hist_base=cv2.calcHist(base,channel,None,hist_size,ranges,accumulate=0)
cv2.normalize(hist_base,hist_base,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)

hist_half=cv2.calcHist(base_half,channel,None, hist_size,ranges,accumulate=0)
cv2.normalize(hist_half,hist_half,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)

hist_1=cv2.calcHist(image_1,channel,None, hist_size,ranges,accumulate=0)
cv2.normalize(hist_1,hist_1,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)

hist_2=cv2.calcHist(image_2,channel,None, hist_size,ranges,accumulate=0)
cv2.normalize(hist_2,hist_2,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)

for compare in range(4):
    base_base=cv2.compareHist(hist_base,hist_base,compare)
    base_basehalf=cv2.compareHist(hist_base,hist_half,compare)
    base_1=cv2.compareHist(hist_base,hist_1,compare)
    base_2=cv2.compareHist(hist_base,hist_2,compare)

    print("compare", compare, "base_base, base_basehalf, base_image 1, base_image2:",base_base,base_basehalf,base_1,base_2)






