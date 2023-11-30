# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 18:04:52 2022

@author: gianl
"""

# image operation
import cv2 as cv
import numpy as np

from ATMANpy.types.Points import Point2D, SafePoint2D
from ATMANpy.types.Lines import Line, Lines, CourtLines, Region
from ATMANpy.types.Frames import Frame, BGRFrame, GreyFrame, BoolFrame, HSVFrame, MaskFrame
from ATMANpy.types.Contours import Contour, Contours


def get_initial_ROIs(schemaLines, 
                     width_factor = 1, depth_factor = 0.5, add_depth_factor_top = 0.5):
    
    width_add = width_factor * schemaLines.doubles_alley_width
    depth_add = depth_factor * schemaLines.side_length
    
    bot_court = schemaLines.get_half_region(side = 'Bottom', width = 'Doubles')
    
    ### make symmetrical ROIs
    ROI_lower = Region.from_points(bot_court['Top'].leftPt, 
                                   bot_court['Top'].rightPt, 
                                   bot_court['Bottom'].leftPt.left(width_add).down(depth_add), 
                                   bot_court['Bottom'].rightPt.right(width_add).down(depth_add))
    
    ROI_upper = Region.from_lines(ROI_lower.transform('flip_vertically', schemaLines['Net Line']))

    ### adjust upper ROI
    depth_add_top = add_depth_factor_top * schemaLines.side_length
    
    top_left, top_right, bot_left, bot_right = ROI_upper.to_points()
    ROI_upper = Region.from_points(top_left.up(depth_add_top), top_right.up(depth_add_top), 
                                   bot_left, bot_right)

    return ROI_lower, ROI_upper

def get_service_ROIs(schemaLines, 
                     width_factor = 1, depth_factor = 0.5, 
                     add_depth_factor_bottom = 0.5,
                     add_depth_factor_top = 0.2):
    
    width_add = width_factor * schemaLines.doubles_alley_width
    depth_add = depth_factor * schemaLines.side_length
    
    bot_court = schemaLines.get_half_region(side = 'Bottom', width = 'Doubles')
    
    ### make symmetrical ROIs
    ROI_lower = Region.from_points(bot_court['Bottom'].leftPt, ### service line
                                   bot_court['Bottom'].rightPt,  # service line
                                   bot_court['Bottom'].leftPt.left(width_add).down(depth_add), 
                                   bot_court['Bottom'].rightPt.right(width_add).down(depth_add))
    
    ROI_upper = Region.from_lines(ROI_lower.transform('flip_vertically', schemaLines['Net Line']))

    ### adjust lower ROI
    depth_add_bottom = add_depth_factor_bottom * schemaLines.side_length
    ### adjust upper ROI
    depth_add_top = add_depth_factor_top * schemaLines.side_length
    
    top_left, top_right, bot_left, bot_right = ROI_lower.to_points()
    ROI_lower = Region.from_points(top_left.up(depth_add_bottom), top_right.up(depth_add_bottom), 
                                   bot_left, bot_right)
    
    top_left, top_right, bot_left, bot_right = ROI_upper.to_points()
    ROI_upper = Region.from_points(top_left.up(depth_add_top), top_right.up(depth_add_top), 
                                   bot_left, bot_right)

    return ROI_lower, ROI_upper

def get_half_court_ROIs(schemaLines, width_factor = 3, depth_factor = 3, overlap_factor = 0.2):
    ### two rois that give simply split the screen in the middle (with some overlap)
    # so that we can use these to limit the adaptive player ROIs.
    # This is to prevent that (by some fluke) the 
    
    width_add = width_factor * schemaLines.doubles_alley_width
    depth_add = depth_factor * schemaLines.side_length

    
    bot_court = schemaLines.get_half_region(side = 'Bottom', width = 'Doubles')
    
    ### make symmetrical ROIs
    ROI_lower = Region.from_points(bot_court['Top'].leftPt.left(width_add), ### service line
                                   bot_court['Top'].rightPt.right(width_add),  # service line
                                   bot_court['Bottom'].leftPt.left(width_add).down(depth_add), 
                                   bot_court['Bottom'].rightPt.right(width_add).down(depth_add))
    
    ROI_upper = Region.from_lines(ROI_lower.transform('flip_vertically', schemaLines['Net Line']))

    ### Slight overlap for the bottom player -> he can appear to be in top court!
    overl_app = overlap_factor * schemaLines.side_length
    
    top_left, top_right, bot_left, bot_right = ROI_lower.to_points()
    ROI_lower = Region.from_points(top_left.up(overl_app), top_right.up(overl_app), 
                                   bot_left, bot_right)

    return ROI_lower, ROI_upper

def get_half_court_Lines(schemaLines, overlap_factor = 0.2):
    ### Slight overlap for the bottom player -> he can appear to be in top court!
    overlap_addon = round(overlap_factor * schemaLines.side_length)
    
    top_halfLine = schemaLines['Net Line']
    bot_halfLine = schemaLines['Net Line'].up(overlap_addon)
    
    return bot_halfLine, top_halfLine
    

def get_initial_ROI_masks(frame, projROI, projCourtLines, which = 'Top'):
    ##### Split regions - upper out, upper in, lower out, lower inside court.
    
    projCourt = projCourtLines.get_half_region(side = which, width = 'Doubles')
    
    roi_mask_out = frame.within(projROI) & ~frame.within(projCourt)
    roi_mask_in = frame.within(projCourt)

    return roi_mask_in, roi_mask_out

# def get_avg_std(cFrm, ROIMask: BoolFrame, white_pixel_threshold = None):
#     if not white_pixel_threshold is None:
#         ### region of interest bool, except we filter out white pixels, too.
#         boolFrame = ROIMask & (cFrm.to_grey() <= white_pixel_threshold)
#     else:
#         boolFrame = ROIMask
    
#     avg = cFrm.mask(boolFrame).sum((0, 1)) / boolFrame.sum((0, 1))
    
#     devFrm = (cFrm - avg) ** 2
#     var = devFrm.mask(boolFrame).sum((0, 1)) / boolFrame.sum((0, 1))
#     std = var ** 0.5
    
#     return avg, std

def get_foreground_ndc_mask(cFrm, avgColor, stdColor, 
                            channel_weights = (1, 1, 1)):
    
    dev = (cFrm - avgColor) / stdColor

    ### deviation is only relevant in region of interest
    # dev = dev.mask(ROIMask) # now done outside of function!
    
    ### calculate ALPHA
    BETA = 1/6 ### quantization step size of H channel ??
    alpha = (0.5 * BETA + stdColor[0]) / stdColor[0]
    
    ### allow for channel thresholds that differ across BGR, or HSV
    channel_thresholds = tuple([alpha * c for c in channel_weights])
    
    ### calculate masks
    background_mask = (dev < channel_thresholds).all(2)
    foreground_mask = ~background_mask
    
    return foreground_mask

def get_line_pixel_mask(frm: Frame, lines: Lines, thickness = 1):
    thickness = round(thickness)
    zeroFrm = frm.zero()
    lineFrm = zeroFrm.put(lines, thickness = thickness)
    return lineFrm > 0

def within_regions(frm, regions):
    if len(regions) == 0:
        return frm
    else:
        r, rs = regions[0], regions[1:]
        return within_regions(frm | frm.within(r), rs)


# def find_player(cFrm: BGRFrame, ROI: Region, 
#                 avgColor_in, stdColor_in, avgColor_out, stdColor_out,
#                 courtLines: Lines, netLines: Lines,
#                 line_width = 8,
#                 court_dilation_factor = 1.5, net_dilation_factor = 2, roi_dilation_factor = 1.5, 
#                 which = 'Top',
#                 verbose = False):

#     hFrm = cFrm.to_HSV()
#     gFrm = cFrm.to_grey()

#     ### Turn region of interest into two masks
#     roi_mask_in, roi_mask_out = get_roi_masks(gFrm, ROI, courtLines, which = which)
    
#     ### Prepare Foreground Masks
#     fore_out_mask = get_ROI_foreground_mask(hFrm, avgColor_out, stdColor_out, roi_mask_out, channel_weights = (1, 0, 1))
#     fore_in_mask = get_ROI_foreground_mask(hFrm, avgColor_in, stdColor_in, roi_mask_in, channel_weights = (1, 0, 1))
#     ndcMask = fore_out_mask | fore_in_mask #non-dominant color
    
#     ### Prepare ROI and Line boundary masks
#     courtline_mask = get_line_pixel_mask(gFrm, courtLines, thickness = court_dilation_factor * line_width)
#     netline_mask = get_line_pixel_mask(gFrm, netLines, thickness = net_dilation_factor * line_width)
#     roi_upper_mask = get_line_pixel_mask(gFrm, ROI, thickness = roi_dilation_factor * line_width)

#     ### Combine ndc mask with negated, dilated courtlines
#     ndcMask = ndcMask & ~(courtline_mask | netline_mask)
#     if verbose: ndcMask.show()
#     # cFrm.mask(ndcMask).show()

#     ########### sobel filter on the image 
#     inFrm = gFrm.mask(roi_mask_in | roi_mask_out)
#     if verbose: inFrm.show()
    
#     sobFrm = inFrm.sobel()
#     avg_in, std_in = get_avg_std(sobFrm, roi_mask_in)
#     avg_out, std_out = get_avg_std(sobFrm, roi_mask_out)
#     sobMask_in = (sobFrm.mask(roi_mask_in) > avg_in + std_in) 
#     sobMask_out = (sobFrm.mask(roi_mask_out) > avg_out + std_out)
#     sobMask = sobMask_in | sobMask_out
#     if verbose: sobMask.show()
    
#     # remove court lines
#     sobMask = sobMask & ~(courtline_mask | netline_mask)
#     # remove edges of area
#     sobMask = sobMask & ~roi_upper_mask
#     if verbose: sobMask.show()
    
#     ### close it, to plug holes
#     kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 11))
#     sobMask = sobMask.to_grey().close(kernel) > 0
#     if verbose: sobMask.show()

#     ### use connected components to remove small noise:
#         # not explained how, but maybe just keep data within those boxes
#         # that are sufficiently large?
#     regList = sobMask.to_grey().connected_regions(size_threshold = 400)
    
#     rectMask = within_regions(sobMask.to_false(), regList)
#     if verbose: rectMask.show()
    
#     sobMask2 = sobMask & rectMask
#     if verbose: sobMask2.show()
    
#     #### Join ndc mask and edge filtered mask
#     outMask = (ndcMask | sobMask) & rectMask
#     if verbose: outMask.show()
    
#     ### open it a bit, to remove salt and pepper
#     kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
#     outMask = outMask.to_grey().open(kernel) > 0
#     if verbose: outMask.show()
    
#     ### reapply connected regions algorithm, to find largest connected region
#     regList = outMask.to_grey().connected_regions(size_threshold = 400)
#     largestRegion = sorted(regList, key = lambda r: r.area)[-1]
    
#     return largestRegion


def get_burn_in_mask(medFrm: Frame, stdFrm: Frame, imgFrm: Frame, H,
                     normalized_deviation_threshold = 1, channel_weights = (1, 1, 1),
                     verbose = False):
    """
    medFrm and stdFrm are the standard deviations of the court color, 
    projected onto the court schema during burn in.
    They are basically what is produced and saved by the burn in script
    """
    if not type(medFrm) == type(stdFrm):
        raise Exception('Median and Stan. Dev. Frames have to be of same type!')
    
    if isinstance(medFrm, HSVFrame):
        try:
            imgFrm = imgFrm.to_HSV()
        except AttributeError:
            pass
    elif isinstance(medFrm, BGRFrame):
        try:
            imgFrm = imgFrm.to_BGR()
        except AttributeError:
            pass
    
    
    # H_inv = np.linalg.inv(H)
    wrpFrm = medFrm.warp(H, cols = imgFrm.cols, rows = imgFrm.rows)
    if verbose: wrpFrm.show()
    
    stdwrpFrm = stdFrm.warp(H, cols = imgFrm.cols, rows = imgFrm.rows)
    if verbose: stdwrpFrm.show()
    
    ### Background Differencing ###
    normdiffFrm = (imgFrm - wrpFrm) / stdwrpFrm.clip_lo(1)
    
    ### exclude everything_outside wrpFrms scope. this way, it will not bleed into the frame if we gaussian blur.
    normdiffFrm = normdiffFrm.mask((wrpFrm > 0).any_channel())
    
    ### Adding a blur stage!
    normdiffFrm = normdiffFrm.gaussian_blur((21, 21))
    
    ### Threshold the normalized difference frame
    backgroundMask = normdiffFrm < [c * normalized_deviation_threshold for c in channel_weights]
    backgroundMask = backgroundMask.all_channels()
    foregroundMask = ~backgroundMask
    
    if verbose:
        pixel_count = (foregroundMask.rows * foregroundMask.cols)
        proportion_foreground = (foregroundMask.sum((0, 1)) / 255) / pixel_count
        print(proportion_foreground)
    
    ### Adding a closing stage
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 18))
    foregroundMask = foregroundMask.close(kernel)
    
    if verbose: imgFrm.mask(foregroundMask).show()
    return foregroundMask


def find_best_expansion(histFrm, previous_expansion = None):
    """
    A function that finds the most parsimonious way to select as much
    probability mass as possible from a histogram frame.
    
    
    Parameters
    ----------
    histFrm : pd.DataFrame
        The histogram frame. Two dimensional -> has H and V channels of normdiffFrm, usually.
    current_expansion : tuple
        The current slice (x, y) into the histogram frame
    Returns
    -------
    a new expansion
    """
    if previous_expansion is None:
        x, y = 0, 0
    else:
        x, y = previous_expansion
    
    additional_mass_from_x = histFrm.loc[x+1,y].sum()
    additional_mass_from_y = histFrm.loc[x,y + 1].sum()
    if additional_mass_from_x > additional_mass_from_y:
        return (x + 1, y)
    else:
        return (x, y + 1)
    
def stop_criterion(histFrm, previous_expansion, current_expansion,
                   minimum_proportion = 0.01, minimum_step_gain = 0.015,
                   verbose = False):    

    x_prev, y_prev = previous_expansion
    x_curr, y_curr = current_expansion
    
    previous_mass = histFrm.loc[:x_prev, :y_prev].sum().sum()
    current_mass = histFrm.loc[:x_curr, :y_curr].sum().sum()
    
    if current_mass - previous_mass < minimum_step_gain:
        if verbose: print(f'Stopping due to step size: {current_mass - previous_mass} < {minimum_step_gain}')
        return True
    
    if current_mass < minimum_proportion:
        if verbose: print(f'Stopping due small remaining mass: {minimum_proportion}')
        return True
    
    return False
    

def get_burn_in_mask_iterative(medFrm: Frame, stdFrm: Frame, imgFrm: Frame, H,
                               minimum_proportion = 0.01, minimum_step_gain = 0.015,
                               verbose = False):
    """
    medFrm and stdFrm are the standard deviations of the court color, 
    projected onto the court schema during burn in.
    They are basically what is produced and saved by the burn in script
    """
    if not type(medFrm) == type(stdFrm):
        raise Exception('Median and Stan. Dev. Frames have to be of same type!')
    
    if isinstance(medFrm, HSVFrame):
        try:
            imgFrm = imgFrm.to_HSV()
        except AttributeError:
            pass
    elif isinstance(medFrm, BGRFrame):
        try:
            imgFrm = imgFrm.to_BGR()
        except AttributeError:
            pass
    
    
    # H_inv = np.linalg.inv(H)
    wrpFrm = medFrm.warp(H, cols = imgFrm.cols, rows = imgFrm.rows)
    if verbose: wrpFrm.show()
    
    stdwrpFrm = stdFrm.warp(H, cols = imgFrm.cols, rows = imgFrm.rows)
    if verbose: stdwrpFrm.show()
    
    ### Background Differencing ###
    normdiffFrm = (imgFrm - wrpFrm) / stdwrpFrm.clip_lo(1)
    
    ### exclude everything_outside wrpFrms scope. this way, it will not bleed into the frame if we gaussian blur.
    normdiffFrm = normdiffFrm.mask((wrpFrm > 0).any_channel())
    
    ### Adding a blur stage!
    normdiffFrm = normdiffFrm.gaussian_blur((21, 21))
    
    ### Iterative discovery of best background parameters
    histFrm = normdiffFrm.hist(['H', 'V'], normalize = True)
    
    previous_expansion = (0, 0)
    current_expansion = find_best_expansion(histFrm, 
                                            previous_expansion = previous_expansion)
    while not stop_criterion(histFrm, previous_expansion, current_expansion,
                             minimum_proportion = minimum_proportion, 
                             minimum_step_gain = minimum_step_gain,
                             verbose = verbose):
        
        previous_expansion = current_expansion
        current_expansion = find_best_expansion(histFrm, 
                                                previous_expansion = previous_expansion)
    else:
        best_expansion = previous_expansion ### since the current expansion triggered the stop criterion.
    
    best_h, best_v = best_expansion
    
    backgroundMask = (normdiffFrm.H() <= best_h) & (normdiffFrm.V() <= best_v)
    foregroundMask = ~backgroundMask
    
    if verbose:
        pixel_count = (foregroundMask.rows * foregroundMask.cols)
        proportion_foreground = (foregroundMask.sum((0, 1)) / 255) / pixel_count
        print(proportion_foreground)
    
    ### Adding a closing stage
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 18))
    foregroundMask = foregroundMask.close(kernel)
    
    if verbose: imgFrm.mask(foregroundMask).show()
    return foregroundMask

    
def get_shadow_mask(medFrm: Frame, imgFrm: Frame, H,
                    verbose = False,
                    h_threshold = 5, s_threshold = 10, 
                    v_threshold_lower = 0.75,  ### the brighter the sun, the starker the shadow, the lower this value has to be.
                    v_threshold_upper = 0.97):
    
    wrpFrm = medFrm.warp(H, cols = imgFrm.cols, rows = imgFrm.rows)
    if verbose: wrpFrm.show()

    diffFrm = imgFrm - wrpFrm

    ### H channel -> affected only slightly!
    H_THRESHOLD = 5
    HBool = diffFrm.H() < H_THRESHOLD

    ### S channel -> affected slightly, but usually lowered!
    S_THRESHOLD = 10
    S_is_lowered = (imgFrm.S() < wrpFrm.S())
    SBool = (diffFrm.S() < S_THRESHOLD) & S_is_lowered
    
    ### V channel -> lowered by 10-20% typically, according to cucchiara et al 2001
    ratioArr = imgFrm.V().array.astype(float) / np.maximum(wrpFrm.V().array.astype(float), 1)
    ratioFrm = Frame(ratioArr)
    VBool = (ratioFrm > v_threshold_lower) & (ratioFrm < v_threshold_upper)
    
    shadowMask = HBool & SBool & VBool 
    
    if verbose: imgFrm.mask(~shadowMask).show()
    
    ### close the candidate shadows, since shadows of a player are contiguous.
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    shadowMask = shadowMask.close(kernel)
    
    if verbose: imgFrm.mask(~shadowMask).show()
    
    return shadowMask
    

def find_player_with_burn_in(hFrm: HSVFrame, burninMask: BoolFrame, 
                             courtLines: Lines, netLines: Lines,
                             initial_inner_avgCol, initial_outer_avgCol,
                             initial_inner_stdCol, initial_outer_stdCol,
                             initialROI, playerROI = None,
                             line_width = 8,
                             court_dilation_factor = 1.5, net_dilation_factor = 2, 
                             verbose = False):
    
    if not isinstance(hFrm, HSVFrame):
        raise Exception('HSV Frame needed to find player')
    
    if verbose:
        cFrm = hFrm.to_BGR()
    
    open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 20))
    foreground_mask_channel_weights = (1, 20, 2)
    
    ### Prepare Foreground Masks
    fore_in_mask = get_foreground_ndc_mask(hFrm, initial_inner_avgCol, initial_inner_stdCol, 
                                           channel_weights = foreground_mask_channel_weights)
    fore_out_mask = get_foreground_ndc_mask(hFrm, initial_outer_avgCol, initial_outer_stdCol, 
                                            channel_weights = foreground_mask_channel_weights)
    ### Mask them with ROI
    ndcMask = (fore_out_mask | fore_in_mask) & hFrm.within(initialROI) #non-dominant color
    if verbose: cFrm.mask(ndcMask).show()
    
    ### Prepare ROI and Line boundary masks
    courtline_mask = get_line_pixel_mask(hFrm.to_grey(), courtLines, thickness = court_dilation_factor * line_width)
    netline_mask = get_line_pixel_mask(hFrm.to_grey(), netLines, thickness = net_dilation_factor * line_width)
    # roi_upper_mask = get_line_pixel_mask(gFrm, ROI, thickness = roi_dilation_factor * line_width)
    
    ### Combine ndc mask with negated, dilated courtlines
    ndcMask = ndcMask & ~(courtline_mask | netline_mask)
    if verbose: cFrm.mask(ndcMask).show()
    
    ### Open and Close NDC mask - first remove noise, then fill gaps
    ndcMask = ndcMask.to_grey().opse(open_kernel, close_kernel) > 0
    if verbose: cFrm.mask(ndcMask).show()
    
    ####### FOREGROUND MASK ####
    if verbose: cFrm.mask(burninMask).show()
    
    burninMask = burninMask.to_grey().opse(open_kernel, close_kernel) > 0
    if verbose: cFrm.mask(burninMask).show()
    
    pMask = ndcMask & burninMask
    if verbose: cFrm.mask(pMask).show()
    
    regList = pMask.to_grey().connected_regions(size_threshold = 400)
    largestRegion = sorted(regList, key = lambda r: r.area)[-1]
    
    if verbose: cFrm.put(largestRegion).show()
    return largestRegion

    
def find_player_only_burn_in(hFrm, burninMask, roiMask, shadowMask = None, 
                             verbose = False):
    SIZE_THRESHOLD = 0
    
    roi_burninMask = burninMask & roiMask
    
    if shadowMask is not None:
        roi_burninMask = roi_burninMask & ~shadowMask
    
    regList = roi_burninMask.to_grey().connected_regions(size_threshold = SIZE_THRESHOLD)
    
    if len(regList) > 0: 
        largestRegion = sorted(regList, key = lambda r: r.area)[-1]
        if verbose: hFrm.put(largestRegion).show()
        return largestRegion
        
def find_player_contour(hFrm, burninMask, roiMask, shadowMask = None, 
                        verbose = False):
    
    MIN_PLAYER_SIZE = 50 ### minimum size of player, in pixels.
    
    roi_burninMask = burninMask & roiMask
    
    if shadowMask is not None:
        roi_burninMask = roi_burninMask & ~shadowMask
        
    cntList, _ = roi_burninMask.to_grey().find_contours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cntList = [Contour(cnt) for cnt in cntList]
    cntList = [cnt for cnt in cntList if cnt.area() > MIN_PLAYER_SIZE]
    
    if len(cntList) > 0:
        largestContour = sorted(cntList, key = lambda c: c.area())[-1]
        if verbose: 
            hFrm.put(largestContour).show()
        return Contour(largestContour)

def calculate_contour_overlap_factor(cnt_interest, cnt_reference, frame_shape):
    LOWER_BOUND_WEIGHT = 0.1
    if cnt_reference is None:
        return 1
    else:
        cntMask_interest = MaskFrame.false(frame_shape).within_contour(cnt_interest) 
        cntMask_reference = MaskFrame.false(frame_shape).within_contour(cnt_reference)
        
        ovlp_sum = (cntMask_interest & cntMask_reference).sum(axes = (0, 1))
        refn_sum = cntMask_reference.sum(axes = (0, 1))
        
        return max(ovlp_sum / refn_sum, LOWER_BOUND_WEIGHT)
    

def find_player_contour_with_overlap(burninMask, roiMask, playerContour_prev, shadowMask = None, 
                                     retr = cv.RETR_EXTERNAL, approx = cv.CHAIN_APPROX_SIMPLE,
                                     verbose = False):
    
    MIN_PLAYER_SIZE = 50 ### minimum size of player, in pixels.
    
    from functools import partial
    
    roi_burninMask = burninMask & roiMask
    
    if shadowMask is not None:
        roi_burninMask = roi_burninMask & ~shadowMask
        
    myCnts, _ = roi_burninMask.to_grey().find_contours(retr, approx)
    
    if myCnts.length() > 0:
        weight_func = partial(calculate_contour_overlap_factor, 
                              frame_shape = burninMask.shape, cnt_reference = playerContour_prev)
        largest_weighted_contour = sorted(myCnts.contourList, key = lambda c: c.area() * weight_func(c))[-1]
        if largest_weighted_contour.area() > MIN_PLAYER_SIZE:
            return largest_weighted_contour


def find_contours(myMask, min_size = 25, retr = cv.RETR_EXTERNAL, approx = cv.CHAIN_APPROX_SIMPLE):
    ### use cv.RETR_EXTERNAL to only get external contours! RETR_TREE also gives black contours that are completely within white areas!
    myCnts, _ = myMask.to_grey().find_contours(retr, approx)
    myCnts = Contours([myCnt for myCnt in myCnts if myCnt.area() > min_size])
    return myCnts

def find_player_candidate_contours_with_overlap(currCnts, candCnts_prev, relative_to = 'other'):
    candCnts = Contours([]) ### the initialized list of contours that actually have a parent -> the candidates
    parLists = []
    for currCnt in currCnts:
        parList = [] ### the list of the parents of the current contour
        
        for par_idx, candCnt_prev in enumerate(candCnts_prev):
            if currCnt.overlap_factor(candCnt_prev, relative_to = relative_to) > 0: # other, because the interpretation is simple: how much of the previous contour is present in the current contour?
                parList.append(par_idx)
        
        candCnts = candCnts.append(currCnt) ### Contours are not mutable...
        parLists.append(parList) ### ... while lists are.
            
    return candCnts, parLists 

    
    
    
    
    
    
    
    
    
    
    

