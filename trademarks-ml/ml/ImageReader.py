from scipy import ndimage
import numpy as np
import scipy.io as sio
import cv2

img = cv2.imread("/Users/greensod/usptoWork/TrademarkRefiles/data/designCodeExtraction/15-designCodes-Images-Structured/01.01.13/85042461.jpg")
img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.expand_dims(img, axis=0)
print(str(np.shape(img)))

img2 = cv2.imread("/Users/greensod/usptoWork/TrademarkRefiles/data/designCodeExtraction/15-designCodes-Images-Structured/01.01.13/85042461.jpg")
img2 = cv2.resize(img2, (224, 224), interpolation=cv2.INTER_CUBIC)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2 = np.expand_dims(img2, axis=0)
print(str(np.shape(img2)))

arr = np.vstack((img, img2));
print(str(np.shape(arr)))

#
# arr = np.array([])
# arr1 = ndimage.imread("/Users/greensod/usptoWork/TrademarkRefiles/data/designCodeExtraction/15-designCodes-Images-Structured/01.01.13/85042461.jpg")
# print(str(np.shape(arr1)))
# arr1 = np.expand_dims(arr1, axis=0)
# print(str(np.shape(arr1)))
#
# arr2 = ndimage.imread("/Users/greensod/usptoWork/TrademarkRefiles/data/designCodeExtraction/15-designCodes-Images-Structured/01.01.13/85042473.jpg")
# arr2 = np.expand_dims(arr2, axis=0)
# print(str(np.shape(arr2)))
#
#
# arr = [arr1, arr2]
# # arr = np.vstack((arr, arr2));
#
# # print(str(arr.shape))
#
# arr = np.asarray(arr);
#
# print(str(np.shape(arr)))
#
# # contents = sio.loadmat("/Users/greensod/usptoWork/TrademarkRefiles/data/designCodeExtraction/miml-image-data/miml data.mat")
# # print(contents)

for i in range(5):
    print(i)