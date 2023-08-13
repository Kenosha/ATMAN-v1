0# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:30:35 2022

@author: gianl
"""

import numpy as np

from dataclasses import dataclass
from ATMANpy.funcs.type_funcs import typecheck, pointcheck
from typing import Optional

@dataclass(frozen = True)
class Point2D:
    """
    A point: tuple of integers, signifying x and y coordinates.
    """   
    coords: tuple
    value: Optional[None] = None
    
    def __post_init__(self):
        #### IMMUTABILITY BREAKS ####
        object.__setattr__(self, 'x', self.coords[0])
        object.__setattr__(self, 'y', self.coords[1])
        #############################  
        
    ### ARITHMETIC ###
    @staticmethod      
    def get_operant(op):
        try: 
            op_x, op_y = op.coords
        except:
            op_x, op_y = op, op
        finally:
            return op_x, op_y    
    
    def __add__(self, y):
        op_x, op_y = Point2D.get_operant(y)
        outCoords = self.coords[0] + op_x, self.coords[1] + op_y
        return Point2D(outCoords)
    def __sub__(self, y):
        op_x, op_y = Point2D.get_operant(y)
        outCoords = self.coords[0] - op_x, self.coords[1] - op_y
        return Point2D(outCoords)
    def __mul__(self, y):
        op_x, op_y = Point2D.get_operant(y)
        outCoords = self.coords[0] * op_x, self.coords[1] * op_y
        return Point2D(outCoords)
    def __floordiv__(self, y):
        op_x, op_y = Point2D.get_operant(y)
        outCoords = self.coords[0] // op_x, self.coords[1] // op_y
        return Point2D(outCoords)
    def __truediv__(self, y): ### this does not guarantee a safe point!
        op_x, op_y = Point2D.get_operant(y)
        outCoords = self.coords[0] / op_x, self.coords[1] / op_y
        return Point2D(outCoords)
    
    ### METHODS ###
    def up(self, points):
        return Point2D((self.x, max(self.y - points, 0)))

    def down(self, points):
        return Point2D((self.x, self.y + points))

    def left(self, points):
        return Point2D((max(self.x - points, 0), self.y))

    def right(self, points):
        return Point2D((self.x + points, self.y))
    
    def distance_to(self, aPt):
        x_dist_64 = int(self.x - aPt.x) ### to prevent overflow when squaring.
        y_dist_64 = int(self.y - aPt.y) ### to prevent overflow when squaring.
        res = (x_dist_64 ** 2 + y_dist_64 ** 2) ** 0.5
        return res
    
    def conform_to(self, image):
        # force the point to be in bounds
        corr_x = min(max(0, self.x), image.cols - 1)
        corr_y = min(max(0, self.y), image.rows - 1)
        return Point2D((corr_x, corr_y))
    
    def project_vertically(self, aLn):
        new_x = self.x
        new_y = aLn.weak_y(self.x)
        if not new_y is None:
            return Point2D((new_x, new_y))
    
    def project_horizontally(self, aLn):
        new_x = aLn.weak_x(self.y)
        new_y = self.y
        if not new_x is None:
            return Point2D((new_x, new_y))
        
    def flip_vertically(self, aLn):
        """
        Flip vertically across a line by projecting onto it and mirroring distance on other side.
        """
        projPt = self.project_vertically(aLn)
        if not projPt is None:
            new_x = self.x
            new_y = projPt.y + (projPt.y - self.y)
            return Point2D((new_x, new_y))
    
    def flip_horizontally(self, aLn):
        """
        Flip horizontally across a line by projecting onto it and mirroring distance on other side.
        """
        projPt = self.project_horizontally(aLn)
        if not projPt is None:
            new_x = projPt.x + (projPt.x - self.x)
            new_y = self.y
            return Point2D((new_x, new_y))
    
    def apply_homography(self, H):
        """
        Apply a 3x3 homography matrix H to a point.        
        """
        _epsilon = 10 ** -5 # a small number, to approximate points at infinity
        hom_coords = np.matmul(H, [self.x, self.y, 1])
        try:
            norm_coords = hom_coords[:2] / hom_coords[2]
        except:
            ### figure out the scale of the first two coords, so we can approximate the point at infinity
            _smaller = max(min(hom_coords[:2]), 1) ### we're only worried about very small coords, hence the maximum condition
            _scale = round(np.log10(_smaller)) + 1
            norm_coords = hom_coords[:2] / (_epsilon ** _scale)
        return Point2D(tuple(norm_coords))
    
    def is_left(self, inObj):
        if isinstance(inObj, Point2D):
            comp = inObj.x
        else:
            comp = inObj
        return self.x < comp
    
    def is_right(self, inObj):
        if isinstance(inObj, Point2D):
            comp = inObj.x
        else:
            comp = inObj
        return self.x > comp
    
    def is_above(self, inObj):
        if isinstance(inObj, Point2D):
            comp = inObj.y
        else:
            comp = inObj
        return self.y < comp
    
    def is_below(self, inObj):
        if isinstance(inObj, Point2D):
            comp = inObj.y
        else:
            comp = inObj
        return self.y > comp

    def round_coords(self):
        x, y = self.coords
        return (round(x), round(y))
    
    def round_point(self):
        return Point2D(self.round_coords(), value = self.value)
        
    def within(self, myReg):
        tl, tr, bl, br = myReg.to_points()
        if self.is_right(tl) and self.is_below(tl) and self.is_left(br) and self.is_above(br):
            return True
        else:
            return False
        
    def bind_below(self, bound):
        x, y = self.coords
        return Point2D((max(x, bound), max(y, bound)))
    
    def bind_above(self, bound):
        x, y = self.coords
        return Point2D((min(x, bound), min(y, bound)))
        
    def find_perpendicular(self):
        #### interpret a point as a vector, and find one (of two) perpendicular vectors
        v1, v2 = self.coords
        return Point2D((-v2, v1))
    
    def rescale(self, scale_factor):
        return (self * scale_factor).round_point()
    
@dataclass(frozen = True)
class SafePoint2D(Point2D):
    """
    A typechecked point
    """
    def __post_init__(self):
        #### TYPE CHECKS ####
        typecheck(self)
        pointcheck(self)
        
        super(SafePoint2D, self).__post_init__()
        #####################  
        
    ### METHODS ###
    def up(self, points):
        if points == int(points):
            return Point2D((self.x, max(self.y - int(points), 0)))
        else:
            raise TypeError("Move point by an integer amount!")

    def down(self, points):
        if points == int(points):
            return Point2D((self.x, self.y + int(points)))
        else:
            raise TypeError("Move point by an integer amount!")

    def left(self, points):
        if points == int(points):
            return Point2D((max(self.x - int(points), 0), self.y))
        else:
            raise TypeError("Move point by an integer amount!")

    def right(self, points):
        if points == int(points):
            return Point2D((self.x + int(points), self.y))
        else:
            raise TypeError("Move point by an integer amount!")


if __name__ == '__main__':
    aPt = Point2D((20, 30))
    
    aPt.rescale(0.3)

    