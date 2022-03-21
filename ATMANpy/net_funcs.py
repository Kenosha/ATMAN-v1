# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:12:58 2022

@author: gianl
"""

from ATMANpy.Lines import Line, Lines

def get_search_distance(courtLines, factor = 0.9):
    ### SEARCH HEIGHT
    ## vinyes: 10% of court length. 
    ### Let's use instead the area right up, but not including to the service line
    ## for this, we need to find the distance between center line and the top serviced line
    ## for the moment, let's take the length of the top service line.
    search_distance = factor * courtLines['Top Center Line'].length()
    search_distance = int(search_distance) + 1
    return search_distance

def get_left_middle_right_search_points(courtLines):
    ### SEARCH WIDTH
    # vinyes: whole width of court, incl. doubles alley. 
    # We'll instead start not at the net posts, but even farther inside, at the singles lines.
    # this is to avoid reading the sponsors printed in the doubles alley part of the net.
    outerleftPt = courtLines['Net Line'].leftPt
    leftPt = courtLines['Net Line'].weak_intersect(courtLines['Top Singles Deuce Line'])
    middlePt = courtLines['Net Line'].weak_intersect(courtLines['Top Center Line'])
    rightPt = courtLines['Net Line'].weak_intersect(courtLines['Top Singles Ad Line'])
    outerrightPt = courtLines['Net Line'].rightPt
    return outerleftPt, leftPt, middlePt, rightPt, outerrightPt
    

def search_points_up(searchPoints, middlePt, gFrm, search_distance = 100):
    """
    Takes searchPoints, and for each of these, 
        searches search_distance points up for the highest intensity pixel in gFrm
    returns a list of x, y pairs, where 
        x is reported relative to a middle point, and
        y is reported relative to the respective net point.
    """
    
    ### ii) evaluate points above the search line, finding the highest intensity value
    relPoints = []
    for aPt in searchPoints:
        intensities = gFrm.array[aPt.y - search_distance: aPt.y, aPt.x]
        max_int_y = intensities.argmax() ### this is calculated from above...
        max_int_y = search_distance - max_int_y ### while this is calculated from below, from the net line.
    
        relative_x = aPt.x - middlePt.x ### x coordinate, relative to center point.
        
        relPoints.append((relative_x, max_int_y))
        
    return relPoints

def find_net_points(courtLines, gFrm, factor = 0.9, points_to_search = 100):
    search_distance = get_search_distance(courtLines, factor = factor)
    _, leftPt, middlePt, rightPt, _ = get_left_middle_right_search_points(courtLines)
    
    innerLine = Line(leftPt, rightPt)
    searchPoints = innerLine.subdivide(points_to_search)
    
    relPoints = search_points_up(searchPoints, middlePt, gFrm, search_distance = search_distance)
    
    return relPoints


def get_connected_elements(inList):
    """
    Takes a list and returns the lowest and the highest point connected to the median

    """
    
    myList = sorted(inList)
    
    median_index = int((len(myList) - 1) / 2)
    median = inList[median_index]
    
    mySet = set(myList)
    
    ### above median
    num_to_check = median + 1
    
    is_connected = True
    while is_connected:
        if num_to_check in mySet:
            num_to_check += 1
        else:
            highest_connected = num_to_check - 1
            is_connected = False
            
    ### below median
    num_to_check = median - 1
    
    is_connected = True
    while is_connected:
        if num_to_check in mySet:
            num_to_check -= 1
        else:
            lowest_connected = num_to_check + 1
            is_connected = False
    
    return lowest_connected, highest_connected

def remove_outlier_net_points(relPoints):
    #### iii) We know that the netline is smooth - therefore, remove outliers!
    # idea: an intensity is an outlier if it is not connected to the median
    # in the sense that all numbers between it and the median are present in the list.
    intList = [y for x, y in relPoints]
    lo, hi = get_connected_elements(intList)
    return [(x, y) for x, y in relPoints if lo <= y <= hi]
    
def cubic_regress(relPoints):
    ### iv) lets now try to estimate a parabola fitting these points
    import statsmodels.api as sm
    
    Y = [y for x, y in relPoints]
    X = [x ** 2 for x, y in relPoints]
    X = sm.add_constant(X)
    
    model = sm.OLS(Y, X)
    results = model.fit()
    b_0, b_1 = results.params
    return b_0, b_1

def make_net_cord_lines(courtLines, b_0, b_1):
    # v) Approximate the parabola with two lines
    # (look at the picture, the net usually does not have much convexity, so two lines are ok.)
    ### so we want two lines to approximate the net - one going from left net post to middle,
    # sloping down, the other going from middle to right, sloping up.

    leftPt, _, middlePt, _, rightPt = get_left_middle_right_search_points(courtLines)
    
    left_x = leftPt.x - middlePt.x
    right_x = rightPt.x - middlePt.x
    
    left_val = b_0 + b_1 * left_x ** 2
    right_val = b_0 + b_1 * right_x ** 2
    
    leftpostLine = Line(leftPt, leftPt.up(left_val), name = 'Left Post')
    rightpostLine = Line(rightPt, rightPt.up(right_val), name = 'Right Post')
    leftnetLine = Line(leftPt.up(left_val), middlePt.up(b_0), name = 'Left Net')
    rightnetLine = Line(rightPt.up(right_val), middlePt.up(b_0), name = 'Right Net')
    
    netLines = Lines([leftpostLine, rightpostLine, leftnetLine, rightnetLine])
    
    return netLines
