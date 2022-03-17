# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:30:35 2022

@author: gianl
"""

import cv2 as cv
import numpy as np

from ATMANpy.Points import Point
from ATMANpy.Lines import Line, Lines
from ATMANpy.Frames import BGRFrame, GreyFrame

if __name__ == '__main__':
    
    ### Point and ArbPoint: Does typechecking work?
    coords = (-1, 2)
    coords = (1.5, 2)
    Point(coords)
    # SafePoint(coords)
    
    coords = (0, 0)
    a = np.zeros((100, 100), dtype = 'uint8')
    
    aFrm = BGRFrame(a)
    aPt = Point(coords)
    
    oFrm = aFrm.put(aPt, radius = 5)
    oFrm.show()
    
    gFrm = GreyFrame(a)
    oFrm2 = gFrm.put(aPt, radius = 5)
    oFrm2.show()
    
    cFrm = gFrm.to_BGR()
    cFrm = cFrm.put(aPt, radius = 5)
    cFrm.show()


    Point((4, 4)).up(10)

    ### typecheck works? ###
    # aFrm = BGRFrame(1)

    ### immutability preserved?
    gFrm = GreyFrame(a)
    print(gFrm.shape)
    cv.cvtColor(gFrm.array, cv.COLOR_GRAY2BGR)
    print(gFrm.shape)
    # gFrm.array = cv.cvtColor(gFrm.array, cv.COLOR_GRAY2BGR)
    # print(gFrm.shape)


    ### Check Line
    aPt = Point((0, 0))
    bPt = Point((100, 100))
    aLn = Line(aPt, bPt)

    bLn = Line(aPt.down(10), bPt.down(5))

    ### Check Lines
    aLns = Lines([aLn, bLn])

    ##### Check Iteration
    for cPt in cFrm:
        print(cPt)
        
    for cLn in aLns:
        print(cLn)

    ### Check multiple dispatch
    a = np.zeros((200, 200), dtype = 'uint8')
    gFrm = GreyFrame(a)
    gFrm.put(aPt, radius = 4).show()
    gFrm.put(aLn).show()


    ##### Test Intersection of Lines
    a = np.zeros((500, 500), dtype = 'uint8')
    gFrm = GreyFrame(a)
    
    aPt = Point((0, 0))
    bPt = Point((300, 100))
    
    ### normal case
    aLn = Line(aPt, bPt)
    bLn = Line(aPt.down(10), bPt.up(40))
    
    lnFrm = gFrm.put(aLn).put(bLn)
    intPt = aLn.weak_intersect(bLn)
    lnFrm.to_BGR().put(intPt, radius = 5).show()
    
    ### parallel case
    bLn = Line(aPt.down(100), bPt.down(100))
    
    lnFrm = gFrm.put(aLn).put(bLn)
    intPt = aLn.weak_intersect(bLn)
    
    if intPt is None:
        lnFrm.to_BGR().show()
    else:
        lnFrm.to_BGR().put(intPt, radius = 5).show()
    
    
    ### parallel vertical case
    aPt = Point((100, 0))
    bPt = Point((100, 200))
    
    aLn = Line(aPt, bPt)
    bLn = Line(aPt.right(100), bPt.right(100))
    
    lnFrm = gFrm.put(aLn).put(bLn)
    intPt = aLn.weak_intersect(bLn)
    
    if intPt is None:
        lnFrm.to_BGR().show()
    else:
        lnFrm.to_BGR().put(intPt, radius = 5).show()
    
    
    #### check projection
    
    ### within normal segment
    aPt = Point((100, 100))
    aLn = Line(aPt.down(200).left(100), aPt.right(300).up(100))
    
    lnFrm = gFrm.to_BGR().put(aLn)
    lnFrm = lnFrm.put(aPt, color = (0, 0, 255), radius = 4, thickness = 2)
    lnFrm.show()
    
    projPt = aPt.project_vertically(aLn)
    p1 = lnFrm.put(projPt, color = (255, 0, 0), radius = 4, thickness = 2)
    p1.show()
    
    projPt = aPt.project_horizontally(aLn)
    p1 = lnFrm.put(projPt, color = (255, 0, 0), radius = 4, thickness = 2)
    p1.show()
    
    ### vertical line segment
    aPt = Point((100, 100))
    aLn = Line(aPt.down(200).right(100), aPt.right(100).down(100))
    
    lnFrm = gFrm.to_BGR().put(aLn)
    lnFrm = lnFrm.put(aPt, color = (0, 0, 255), radius = 4, thickness = 2)
    lnFrm.show()
    
    projPt = aPt.project_vertically(aLn)
    if not projPt is None:
        p1 = lnFrm.put(projPt, color = (255, 0, 0), radius = 4, thickness = 2)
        p1.show()
    
    projPt = aPt.project_horizontally(aLn)
    p1 = lnFrm.put(projPt, color = (255, 0, 0), radius = 4, thickness = 2)
    p1.show()
    
    ### check flips
    aPt = Point((100, 100))
    aLn = Line(aPt.down(200).left(100), aPt.right(100).down(10))
    
    lnFrm = gFrm.to_BGR().put(aLn)
    lnFrm = lnFrm.put(aPt, color = (0, 0, 255), radius = 4, thickness = 2)
    lnFrm.show()
    
    projPt = aPt.flip_vertically(aLn)
    p1 = lnFrm.put(projPt, color = (255, 0, 0), radius = 4, thickness = 2)
    p1.show()
    
    projPt = aPt.flip_horizontally(aLn)
    p1 = lnFrm.put(projPt, color = (255, 0, 0), radius = 4, thickness = 2)
    p1.show()
    
    
    ## line flips...
    aPt = Point((100, 100))
    aLn = Line(aPt.down(200).left(100), aPt.right(100).down(10))
    bLn = Line(aPt, aPt.up(50).right(25))
    
    lnFrm = gFrm.to_BGR().put(aLn).put(bLn, color = (0, 0, 255))
    lnFrm.show()
    
    p1 = lnFrm.put(bLn.flip_vertically(aLn))
    p1.show()
    
    
    
    
    
    ### check distance
    aPt = Point((100, 0))
    bPt = Point((200, 100))
    
    aPt.distance_to(bPt)
    