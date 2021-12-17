import cv2
import numpy as np
import os

#calculate pHash first
#aHash
needle=cv2.imread("../data/Compare/Alyson_Hannigan_200512.jpg", cv2.IMREAD_GRAYSCALE)
needle=cv2.resize(needle,(8,8))
convert_needle=np.float32(needle)
#print(convert_needle)
mean=np.mean(convert_needle)
#print("mean value:", np.mean(convert_needle))
#dst_needle=cv2.dct(convert_needle)
#dst_needle_flat=dst_needle.flatten()
#dst_needle_flat_no_first=dst_needle_flat[1:]
#mean_dct= sum(dst_needle_flat_no_first)/len(dst_needle_flat_no_first)
print(mean)
#print(dst_needle_flat)
#print(sum(dst_needle_flat))
#print(len(dst_needle_flat))
#print("mean value:", np.mean(dst_needle))

dct=[]
diff=needle[:,:]>mean
hash=sum([2**i for(i,v) in enumerate(diff.flatten()) if v])

print(hash)