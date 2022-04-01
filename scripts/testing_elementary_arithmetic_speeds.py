# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:26:46 2022

@author: gianl
"""

import cv2 as cv
import numpy as np

img1=np.uint8(np.random.randint(0, 255, (480, 640)))
img2=np.uint8(np.random.randint(0, 255, (480, 640)))

def differenceImageV1(img1, img2):
  diff=np.empty_like(img1)
  h, w=img1.shape
  for y in range(h):
    for x in range(w):
      if img1[y, x]<img2[y, x]: diff[y, x]=img2[y, x]-img1[y, x]
      else:                     diff[y, x]=img1[y, x]-img2[y, x]
  return(diff)

def differenceImageV2(img1, img2):
  return(np.uint8(np.abs(np.int16(img1)-img2)))

def differenceImageV3(img1, img2):  # fast - but wrong result
  return(img1-img2)

def differenceImageV4(img1, img2):
  return np.where(img1>img2, img1-img2, img2-img1)

def differenceImageV5(img1, img2):
  a = img1-img2
  b = img2-img1
  c = img1>img2
  return a*c + b*(~c)

def differenceImageV6(img1, img2):
  a = img1-img2
  b = np.uint8(img1<img2) * 254 + 1
  return a * b

def differenceImageV7(img1, img2):
    return np.absolute(img1.astype(float) - img2.astype(float)).astype(np.uint8)


def differenceImageV8(img1, img2):
    return cv.absdiff(img1, img2)

import timeit
def testit():
    for fn in [differenceImageV2, differenceImageV3, differenceImageV4, 
               differenceImageV5, differenceImageV6, differenceImageV7,
               differenceImageV8]:
        print(fn.__name__, np.sum(fn(img1, img2).astype('int64')))
        print(timeit.timeit("%s(img1, img2)" % fn.__name__, "from testing_elementary_arithmetic_speeds import img1, img2, %s" % fn.__name__, number=1000))
        print('')
     
### Channel subtraction

img1=np.uint8(np.random.randint(0, 255, (480, 640, 3)))
img2=np.uint8(np.random.randint(0, 255, (480, 640, 3)))
     
chArr = [36, 2, 10]

outArr = np.stack([cv.absdiff(img1[:, :, chIdx], chVal) for chIdx, chVal in enumerate(chArr)], -1)
outArr.shape



def diff1(img, chArr):    
    cv.absdiff(img1, int(chArr[0]))
c


        
if __name__ == '__main__':
    testit()