"""
These functions take an Abstract object (from ATMANpy.types.Abstract) and keywords, 
and visualize the Abstract object in a certain way, returning a View
"""

import numpy as np

from ATMANpy.types.Lines import CourtLines
from ATMANpy.types.Visual3D import Scene, DummyScene, Camera, View

import ATMANpy.auxil.constants as const

from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Shapes3D import Line3D


def interpolate(startCam, endCam, num_frames):
    assert startCam.sensor_dims == endCam.sensor_dims
    frames = range(0, num_frames)
    num_frames = len(frames)

    itpl_factors = np.linspace(0, 1, num_frames, endpoint = True)
    itpl_factors_brdcst = itpl_factors[:, np.newaxis, np.newaxis]
    M_ints = startCam.M_int[np.newaxis] * (1 - itpl_factors_brdcst) + endCam.M_int[np.newaxis] * itpl_factors_brdcst
    M_exts = startCam.M_ext[np.newaxis] * (1 - itpl_factors_brdcst) + endCam.M_ext[np.newaxis] * itpl_factors_brdcst

    itplCams = [Camera(M_int, M_ext, startCam.sensor_dims) for (M_int, M_ext) in zip(M_ints, M_exts)]
    return itplCams

def animate(shapes = None, 
            start_cam = None, stop_cam = None, num_frames = None,
            background_scene = None, previous_foreground_scene = None, dummy_scene = None,
            background_resolution = 1, foreground_resolution = 1, dummy_resolution = 1, 
            cumulative_foreground_transparence = 0, 
            text_lines = None, text_size = 1, text_color = const.BGR['strong_pink'], text_where = (50, 50), text_offset = 50):
    if shapes is None:
        shapes = []

    if num_frames is None:
        num_frames = len(shapes)
    else: ### TODO need a way of interpolating shapes to length of duration
        if len(shapes) == 0:
            constant_shape = None
        elif len(shapes) == 1:
            constant_shape = shapes[0]
        else:
            raise Exception('Not implemented: need a way of interpolating shapes to length of duration... in the meantime, if a duration is given, we have a single shape!')

    ### Create the List of Cameras to be Used
    if start_cam is None:
        start_cam = Camera.calibrated_AO_behind()
    if stop_cam is None:
        stop_cam = Camera.calibrated_AO_behind()
    
    ### Initialize background and foreground
    base_scene = Scene.make_empty()
    if background_scene is not None:
        base_scene = base_scene.combine(background_scene)
    front_scene = Scene.make_empty()
    if previous_foreground_scene is not None:
        front_scene = front_scene.combine(previous_foreground_scene)
    
    cams = interpolate(start_cam, stop_cam, num_frames) 

    frameList = []
    for idx in range(num_frames):
        current_cam = cams[idx]

        try:
            current_shape = constant_shape
        except NameError:
            current_shape = shapes[idx]

        if cumulative_foreground_transparence > 0:
            front_scene = front_scene.transparify(cumulative_foreground_transparence)
            
        front_scene = front_scene.put(current_shape)

        front_view = View.make(current_cam, front_scene, resolution = foreground_resolution)
        base_view = View.make(current_cam, base_scene, resolution = background_resolution)
        current_view = front_view.combine(base_view)

        if dummy_scene is not None:
            dummy_view = View.make(current_cam, dummy_scene, resolution = dummy_resolution)
            current_view = current_view.combine(dummy_view) ### can not combine Scene and DummyScene directly! need to combine views

        myFrm = current_view.snap()

        if text_lines is not None:
            pos_x, pos_y = text_where
            for textStr in text_lines:
                myFrm = myFrm.put(textStr, where = (pos_x, pos_y), color = text_color, size = text_size)
                pos_y += text_offset

        frameList.append(myFrm)

    myFrms = BGRFrames(np.array([f.array for f in frameList]))

    return myFrms, front_scene ### the last loop's current scene may be the next animation's previous_foreground_scene

def animate_static(shape = None, num_frames = 100,
                    static_cam = None, 
                    background_scene = None, previous_foreground_scene = None, dummy_scene = None,
                    background_resolution = 1, foreground_resolution = 1, dummy_resolution = 1, 
                    cumulative_foreground_transparence = 0, 
                    text_lines = None, text_size = 1, text_color = const.BGR['strong_pink'], text_where = (50, 50), text_offset = 50):
    if static_cam is None:
        static_cam = Camera.calibrated_AO_behind()

    ### CREATE BACKGROUND VIEW    
    base_scene = Scene.make_empty()
    if background_scene is not None:
        base_scene = base_scene.combine(background_scene)
    base_view = View.make(static_cam, base_scene, resolution = background_resolution)

    ### CREATE FOREGROUND SCENE
    if shape is None:
        front_scene = Scene.make_empty()
    else:
        front_scene = Scene.make_empty().put(shape)
    if previous_foreground_scene is not None:
        front_scene = front_scene.combine(previous_foreground_scene)
    front_view = View.make(static_cam, front_scene, resolution = foreground_resolution)

    ### CREATE DUMMY VIEW
    if dummy_scene is not None:
        dummy_view = View.make(static_cam, dummy_scene, resolution = dummy_resolution)
    else:
        dummy_view = View.make_empty()

    ### JOIN VIEWS AND SNAP
    static_view = front_view.combine(base_view).combine(dummy_view) ### can not combine Scene and DummyScene directly! need to combine views
    static_frame = static_view.snap()

    ### MANAGE TEXT
    if text_lines is not None:
        pos_x, pos_y = text_where
        for textStr in text_lines:
            static_frame = static_frame.put(textStr, where = (pos_x, pos_y), color = text_color, size = text_size)
            pos_y += text_offset

    ### COLLATE STATIC FRAMES 
    frameList = [static_frame for _ in range(num_frames)]
    myFrms = BGRFrames(np.array([f.array for f in frameList]))
    
    return myFrms, front_scene

def divide_line(someLn, num_divisions, reverse_direction = False):
    if someLn is None:
        itpl_lines = [None] * num_divisions
    else:
        leftPt, rightPt = someLn.leftPt, someLn.rightPt

        flightVec = someLn.to_vector()
        
        itpl_steps = np.linspace(0, 1, endpoint = True, num = num_divisions + 1)
        
        if not reverse_direction: ### left to right
            itpl_points = [leftPt + flightVec * step for step in itpl_steps]
        else:
            itpl_points = [rightPt - flightVec * step for step in itpl_steps]

        itpl_lines = []
        prevPt = None
        for currPt in itpl_points:
            if prevPt is not None:
                itplLn = Line3D(prevPt, currPt, color = someLn.color, thickness = someLn.thickness, alpha = someLn.alpha)
                itpl_lines.append(itplLn)
            prevPt = currPt
    return itpl_lines


def kde_heatmap(myShots, showReg = None,
                position_func = 'get_first_bounce_position',
                bandwidth = 30,
                zero_color = (0, 0, 0), one_color = const.BGR['intense_red']):
    ### Kernel Density Heatmap with first bounce positions, pictured within a region

    ### Define our Kernel Density Estimator
    def epanechnikov(posArr_nonnan, showArr, bandwidth = 30):
        
        diffArr = showArr[np.newaxis] - posArr_nonnan[:, np.newaxis, np.newaxis]
        norm_diffArr = diffArr / bandwidth
        norm_devArr = (norm_diffArr ** 2).sum(axis = -1)
        
        kde_values_for_each_kernel = (1 - norm_devArr) * (norm_devArr < 1)
        kde_values = kde_values_for_each_kernel.mean(axis = 0)
        
        kde_values_renorm = kde_values / kde_values.max()
        
        bgr = np.array([zero_color, one_color]) ### bgr for kde = 0 and 1 resp.
        alpha = np.array([[0], [1]]) ### alpha for kde = 0 and 1 resp.
        
        kde_brdcst = kde_values_renorm[:, :, np.newaxis]
        bgr_brdcst = bgr[:, np.newaxis, np.newaxis, :]
        alpha_brdcst = alpha[:, np.newaxis, np.newaxis, :]
        
        bgrArr = (1 - kde_brdcst) * bgr_brdcst[0] + kde_brdcst * bgr_brdcst[1]
        alphaArr = (1 - kde_brdcst) * alpha_brdcst[0] + kde_brdcst * alpha_brdcst[1]
        return bgrArr, alphaArr
    
    ### Gather courtlines
    if showReg is None:
        showReg = CourtLines.make().get_court_region()
    
    ### Get first bounce position
    coords = getattr(myShots, position_func)()
    posArr = np.array(coords, dtype = 'float32')[:, :2]
    posArr_nonnan = posArr[~np.isnan(posArr).all(axis = 1)]
    
    ### delineate the region to show
    x1, y1 = showReg.p1.coords
    x2, y2 = showReg.p4.coords
    xv, yv = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    showArr = np.stack([xv, yv], axis = -1)
    num_x, num_y, _ = showArr.shape
    
    ### Prepare the resulting Scene
    bgrArr, alphaArr = epanechnikov(posArr_nonnan, showArr, bandwidth = bandwidth)
    bgrArr_long = bgrArr.reshape((num_x * num_y, 3))
    alphaArr_long = alphaArr.reshape((num_x * num_y, 1))

    showArr_long = showArr.reshape((num_x * num_y, 2))
    zArr = np.zeros((num_x * num_y, 1))
    showArr_long = np.concatenate([showArr_long, zArr], axis = 1)
    
    ### Visualize
    kdeScn = DummyScene([showArr_long], [bgrArr_long], [alphaArr_long])
    # courtScn = Scene.make_court()
    # backView = View.make(myCam, courtScn, resolution = resolution)
    # kdeView = View.make(myCam, kdeScn)
    # combView = backView.combine(kdeView)
    
    return kdeScn