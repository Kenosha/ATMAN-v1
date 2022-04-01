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
   
if __name__ == '__main__':
    testit()
    print('_____________')
    
### Channel subtraction

img3=np.uint8(np.random.randint(0, 255, (480, 640, 3)))
     
chArr = [36, 2, 10]

def subtractChannels1(img, channels):
    return np.stack([cv.absdiff(img[:, :, chIdx], chVal) for chIdx, chVal in enumerate(chArr)], -1)

def subtractChannels2(img, channels):
    return cv.merge([cv.absdiff(img[:, :, chIdx], chVal) for chIdx, chVal in enumerate(chArr)])

def subtractChannels3(img, channels):
    return cv.merge([cv.absdiff(chImg, chVal) for chImg, chVal in zip(cv.split(img), channels)])

def subtractChannels4(img, channels):
    return np.absolute(img - channels)

def testit():
    for fn in [subtractChannels1, subtractChannels2, subtractChannels3, subtractChannels4]:
        print(fn.__name__, np.sum(fn(img3, chArr).astype('int64')))
        print(timeit.timeit("%s(img3, chArr)" % fn.__name__, "from testing_elementary_arithmetic_speeds import img3, chArr, %s" % fn.__name__, number=1000))
        print('')

if __name__ == '__main__':
    testit()
    print('_____________')
    
    
### Channel Comparison

img3=np.uint8(np.random.randint(0, 255, (480, 640, 3)))
     
chArr = [36, 2, 10]

def compareChannels1(img, channels):
    return cv.merge([cv.compare(chImg, chVal, cv.CMP_LT)  for chImg, chVal in zip(cv.split(img), channels)])

def compareChannels2(img, channels):
    return img < channels

def testit():
    for fn in [compareChannels1, compareChannels2]:
        print(fn.__name__, np.sum(fn(img3, chArr).astype('int64')))
        print(timeit.timeit("%s(img3, chArr)" % fn.__name__, "from testing_elementary_arithmetic_speeds import img3, chArr, %s" % fn.__name__, number=1000))
        print('')

if __name__ == '__main__':
    testit()
    print('_____________')
    
    
 
### All and Any

img3=np.uint8(np.random.randint(0, 255, (480, 640, 3)))
     
chArr = [36, 2, 10]

boolFrm = compareChannels1(img3, chArr)

def any1(boolImg):
    return boolFrm.any(2)

def any2(boolImg):
    r, g, b = cv.split(boolImg)
    return cv.bitwise_or(cv.bitwise_or(r, g), b)
    
    
def testit():
    for fn in [any1, any2]:
        print(fn.__name__, np.sum(fn(boolFrm).astype('int64')))
        print(timeit.timeit("%s(boolFrm)" % fn.__name__, "from testing_elementary_arithmetic_speeds import boolFrm, %s" % fn.__name__, number=1000))
        print('')

if __name__ == '__main__':
    testit()
    print('_____________')

    
    