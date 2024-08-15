"""
These functions take an Abstract object (from ATMANpy.types.Abstract) and keywords, 
and visualize the Abstract object in a certain way, returning a View
"""

import numpy as np

from ATMANpy.types.Lines import CourtLines
from ATMANpy.types.Visual3D import Scene, DummyScene, Camera, View

import ATMANpy.auxil.constants as const

from ATMANpy.types.MultiFrames import BGRFrames
from ATMANpy.types.Shapes3D import Point3D, Line3D

from ATMANpy.types.MultiFrames import Frames

def blinking(onFrm, offFrm, blinks_per_second = 1, duration_in_frames = 100):
    if isinstance(onFrm, Frames):
        onFrm = onFrm[-1]
    if isinstance(offFrm, Frames):
        offFrm = offFrm[-1]

    hold_frames = 25 // blinks_per_second
    assert hold_frames > 0

    onFrms = BGRFrames.from_frame(onFrm, hold_frames)
    offFrms = BGRFrames.from_frame(offFrm, hold_frames)
    combFrms = BGRFrames.from_frames([onFrms, offFrms])

    repeats = duration_in_frames // (hold_frames * 2) + 1

    return BGRFrames.from_frames([combFrms] * repeats)[:duration_in_frames]

def interpolate_camera(startCam, endCam, num_frames):
    assert startCam.sensor_dims == endCam.sensor_dims
    frames = range(0, num_frames)
    num_frames = len(frames)

    itpl_factors = np.linspace(0, 1, num_frames, endpoint = True)
    itpl_factors_brdcst = itpl_factors[:, np.newaxis, np.newaxis]
    M_ints = startCam.M_int[np.newaxis] * (1 - itpl_factors_brdcst) + endCam.M_int[np.newaxis] * itpl_factors_brdcst
    M_exts = startCam.M_ext[np.newaxis] * (1 - itpl_factors_brdcst) + endCam.M_ext[np.newaxis] * itpl_factors_brdcst

    itplCams = [Camera(M_int, M_ext, startCam.sensor_dims) for (M_int, M_ext) in zip(M_ints, M_exts)]
    return itplCams

def interpolate_shape(fs_shape, ls_shape, itpl_lambda = 0.5):
    assert fs_shape.__class__ == ls_shape.__class__
    return fs_shape.interpolate(ls_shape, itpl_lambda = itpl_lambda)

def interpolate_scenes(start_scene, end_scene, num_frames):
    start_shapes, end_shapes = start_scene.shapes, end_scene.shapes
    assert len(start_shapes) == len(end_shapes)

    scenes = []
    for frame in range(num_frames):
        lmbd = frame / (num_frames - 1)
        itpl_shapes = [interpolate_shape(fs, ls, itpl_lambda = lmbd) for fs, ls in zip(start_shapes, end_shapes)]
        scenes.append(Scene(itpl_shapes))

    return scenes

def interpolate_dummy_scenes(start_dummyScn, end_dummyScn, num_frames,
                             x_fill = None, y_fill = None, z_fill = None):
    def fill_column(arr1, arr2, fill_val, which):
        ### which -> 0, 1 or 2
        if isinstance(fill_val, int):
            arr1[:, which] = np.where(np.isnan(arr1[:, which]), fill_val, arr1[:, which])
            arr2[:, which] = np.where(np.isnan(arr2[:, which]), fill_val, arr2[:, which])
        elif isinstance(fill_val, str):
            if fill_val == 'other':
                arr1[:, which] = np.where(np.isnan(arr1[:, which]), arr2[:, which], arr1[:, which])
                arr2[:, which] = np.where(np.isnan(arr2[:, which]), arr1[:, which], arr2[:, which])
        return arr1, arr2
    
    fs_len, ls_len = len(start_dummyScn.rastArrs), len(end_dummyScn.rastArrs)
    assert fs_len == ls_len

    scenes = []
    for frame in range(num_frames):
        lmbd = frame / (num_frames - 1)
        
        itpl_rastArrs = []
        itpl_colArrs = []
        itpl_alphaArrs = []
        for idx in range(fs_len):
            fs_rastArr, ls_rastArr = start_dummyScn.rastArrs[idx].copy(), end_dummyScn.rastArrs[idx].copy()
            fs_colArr, ls_colArr = start_dummyScn.colArrs[idx].copy(), end_dummyScn.colArrs[idx].copy()
            fs_alphaArr, ls_alphaArr = start_dummyScn.alphaArrs[idx].copy(), end_dummyScn.alphaArrs[idx].copy()

            ### figure out how to fill nans in rastArr
            fs_rastArr, ls_rastArr = fill_column(fs_rastArr, ls_rastArr, x_fill, 0)
            fs_rastArr, ls_rastArr = fill_column(fs_rastArr, ls_rastArr, y_fill, 1)
            fs_rastArr, ls_rastArr = fill_column(fs_rastArr, ls_rastArr, z_fill, 2)

            itpl_rastArr = (1 - lmbd) * fs_rastArr + lmbd * ls_rastArr
            itpl_colArr = (1 - lmbd) * fs_colArr + lmbd * ls_colArr
            itpl_alphaArr = (1 - lmbd) * fs_alphaArr + lmbd * ls_alphaArr

            itpl_rastArrs.append(itpl_rastArr)
            itpl_colArrs.append(itpl_colArr)
            itpl_alphaArrs.append(itpl_alphaArr)

        frameScn = DummyScene(itpl_rastArrs, itpl_colArrs, itpl_alphaArrs)
        scenes.append(frameScn)

    return scenes

def animate_sequence(animation_sequence_objects, annotated_points = None,
                    background_scene = None, previous_foreground_scene = None, previous_dummy_scene = None, previous_annotated_points = None,
                    argument_resolution = 1, background_resolution = 1, foreground_resolution = 1, dummy_resolution = 1, 
                    argument_luminosity = 1, adaptive_resolution_exponent = 0, 
                    cumulate_argument = False, cumulative_foreground_transparence = None, 
                    argument_blinks_per_second = None,
                    start_cam = None, stop_cam = None):
    ### animation_sequence_objects: a list of shapes, scenes or dummy scenes that are animated in sequence -> one list element per frame
    ### other scenes are static/previous scenes that are not animated, but can be faded out/in
    num_frames = len(animation_sequence_objects)

    ### Create the List of Cameras to be Used
    if start_cam is None:
        start_cam = Camera.calibrated_AO_behind()
    if stop_cam is None:
        stop_cam = Camera.calibrated_AO_behind()
    
    ### Initialize background and foreground
    if background_scene is None:
        background_scene = Scene.make_empty()
   
    if previous_foreground_scene is None:
        previous_foreground_scene = Scene.make_empty()

    if previous_dummy_scene is None:
        previous_dummy_scene = DummyScene.make_empty()
    
    ### create joint_annotated_points -> previous points and current foreground point
    if annotated_points is None: annotated_points = []
    if previous_annotated_points is None: previous_annotated_points = []

    ### cull previous annotated points if they are included in annotated points (this can be useful if one wants to keep a text lit up)
    previous_annotated_points = [p for p in previous_annotated_points if p not in annotated_points]

    ### change text luminosity 
    argument_annotated_points_brightened = [p.dim(argument_luminosity) for p in annotated_points]
    argument_annotated_points_dimmed = [p.dim(1 / argument_luminosity) for p in annotated_points] ### this passes into the next_animation_dictionary
    annotated_points_to_show = [*argument_annotated_points_brightened, *previous_annotated_points]

    cams = interpolate_camera(start_cam, stop_cam, num_frames) 

    frameList = []
    for idx in range(num_frames):
        current_cam = cams[idx]

        ### fade previous foreground scenes
        if cumulative_foreground_transparence is not None:
            previous_foreground_scene = previous_foreground_scene.transparify(cumulative_foreground_transparence)
            previous_dummy_scene = previous_dummy_scene.transparify(cumulative_foreground_transparence)
              
        ### create current view
        current_object = animation_sequence_objects[idx]
        if isinstance(current_object, DummyScene) or isinstance(current_object, Scene):
            current_scene = current_object
        elif isinstance(current_object, list):
            current_scene = Scene(current_object)
        else:
            current_scene = Scene([current_object])

        current_scene_brightened = current_scene.dim(argument_luminosity)
        current_scene_dimmed = current_scene.dim(1 / argument_luminosity)
        current_view = View.make(current_cam, current_scene_brightened, annotatedPoints = annotated_points_to_show, resolution = argument_resolution, adaptive_resolution_exponent = adaptive_resolution_exponent) 
    
        ### create previous and base views
        front_view = View.make(current_cam, previous_foreground_scene, resolution = foreground_resolution, adaptive_resolution_exponent = adaptive_resolution_exponent)
        dummy_view = View.make(current_cam, previous_dummy_scene, resolution = dummy_resolution) ### TODO: adaptive resolution not done for dummy scenes
        base_view = View.make(current_cam, background_scene, resolution = background_resolution) ### TODO: adaptive resolution redundant for the background?

        joint_view = base_view.combine(dummy_view).combine(front_view).combine(current_view)
        
        myFrm = joint_view.snap()

        frameList.append(myFrm)

        if cumulate_argument: ### this allows the animation_sequence_objects (the argument of the function) to be faded out in trailing motion.
            previous_foreground_scene = previous_foreground_scene.combine(current_scene_brightened, keep_which = 'other')

    myFrms = BGRFrames(np.array([f.array for f in frameList]))

    ### make a kwarg dictionary for the next animation
    if isinstance(current_scene, Scene):
        next_animation_dictionary = {'previous_foreground_scene': previous_foreground_scene.combine(current_scene_dimmed, keep_which = 'other'), ### TODO: might have to be adapted to the case in which cumulate_argument = True
                                     'previous_dummy_scene': previous_dummy_scene,
                                     'previous_annotated_points': [*argument_annotated_points_dimmed, *previous_annotated_points]}
    elif isinstance(current_scene, DummyScene):
        next_animation_dictionary = {'previous_foreground_scene': previous_foreground_scene,
                                     'previous_dummy_scene': previous_dummy_scene.combine(current_scene_dimmed, keep_which = 'other'),
                                     'previous_annotated_points': [*argument_annotated_points_dimmed, *previous_annotated_points]}

    return myFrms, next_animation_dictionary


def animate_cam_motion(animation_object, annotated_points = None, num_frames = None,
                        background_scene = None, previous_foreground_scene = None, previous_dummy_scene = None, previous_annotated_points = None, ### these are static/previous scenes that are not animated, but can be faded out/in
                        argument_resolution = 1, background_resolution = 1, foreground_resolution = 1, dummy_resolution = 1, 
                        argument_luminosity = 1, cumulative_foreground_transparence = None, adaptive_resolution_exponent = 0,
                        argument_blinks_per_second = None,
                        start_cam = None, stop_cam = None):
    ### animation_object: a static shape, list of shapes, scene, or dummyscene. 
    num_frames = int(num_frames)

    ### create current scene
    if animation_object is None:
        static_scene = Scene.make_empty()
    elif isinstance(animation_object, DummyScene) or isinstance(animation_object, Scene):
        static_scene = animation_object
    elif isinstance(animation_object, list):
        static_scene = Scene(animation_object)
    else:
        static_scene = Scene([animation_object])

    ### Create the List of Cameras to be Used
    if start_cam is None:
        start_cam = Camera.calibrated_AO_behind()
    if stop_cam is None:
        stop_cam = Camera.calibrated_AO_behind()
    
    ### Initialize background and foreground
    if background_scene is None:
        background_scene = Scene.make_empty()
   
    if previous_foreground_scene is None:
        previous_foreground_scene = Scene.make_empty()

    if previous_dummy_scene is None:
        previous_dummy_scene = DummyScene.make_empty()
    
    ### dim or lumify scenes based on whether they're argument or previous
    static_scene_brightened = static_scene.dim(argument_luminosity)
    static_scene_dimmed = static_scene.dim(1 / argument_luminosity) ### this passes into the next_animation_dictionary

    ### create joint_annotated_points -> previous points and current foreground point
    if annotated_points is None: annotated_points = []
    if previous_annotated_points is None: previous_annotated_points = []

    ### cull previous annotated points if they are included in annotated points (this can be useful if one wants to keep a text lit up)
    previous_annotated_points = [p for p in previous_annotated_points if p not in annotated_points]

    ### change text luminosity 
    argument_annotated_points_brightened = [p.dim(argument_luminosity) for p in annotated_points]
    argument_annotated_points_dimmed = [p.dim(1 / argument_luminosity) for p in annotated_points] ### this passes into the next_animation_dictionary
    annotated_points_to_show = [*argument_annotated_points_brightened, *previous_annotated_points]

    ### make cams
    cams = interpolate_camera(start_cam, stop_cam, num_frames) 

    frameList = []
    for idx in range(num_frames):
        current_cam = cams[idx]

        ### fade previous foreground scenes
        if cumulative_foreground_transparence is not None:
            previous_foreground_scene = previous_foreground_scene.transparify(cumulative_foreground_transparence)
            previous_dummy_scene = previous_dummy_scene.transparify(cumulative_foreground_transparence)
        
        current_view = View.make(current_cam, static_scene_brightened, annotatedPoints = annotated_points_to_show, resolution = argument_resolution, adaptive_resolution_exponent = adaptive_resolution_exponent) 
    
        ### create previous and base views
        front_view = View.make(current_cam, previous_foreground_scene, resolution = foreground_resolution, adaptive_resolution_exponent = adaptive_resolution_exponent)
        dummy_view = View.make(current_cam, previous_dummy_scene, resolution = dummy_resolution)
        base_view = View.make(current_cam, background_scene, resolution = background_resolution)

        joint_view = base_view.combine(dummy_view).combine(front_view).combine(current_view)
        
        myFrm = joint_view.snap()

        frameList.append(myFrm)

    myFrms = BGRFrames(np.array([f.array for f in frameList]))

    ### make a kwarg dictionary for the next animation
    if isinstance(static_scene, Scene):
        next_animation_dictionary = {'previous_foreground_scene': previous_foreground_scene.combine(static_scene_dimmed, keep_which = 'other'),
                                     'previous_dummy_scene': previous_dummy_scene,
                                     'previous_annotated_points': [*argument_annotated_points_dimmed, *previous_annotated_points]}
    elif isinstance(static_scene, DummyScene):
        next_animation_dictionary = {'previous_foreground_scene': previous_foreground_scene,
                                     'previous_dummy_scene': previous_dummy_scene.combine(static_scene_dimmed, keep_which = 'other'),
                                     'previous_annotated_points': [*argument_annotated_points_dimmed, *previous_annotated_points]}

    return myFrms, next_animation_dictionary


def animate_static(animation_object, annotated_points = None, num_frames = 100, static_cam = None, 
                    background_scene = None, previous_foreground_scene = None, previous_dummy_scene = None, previous_annotated_points = None,
                    argument_resolution = 1, background_resolution = 1, foreground_resolution = 1, dummy_resolution = 1, 
                    argument_luminosity = 1, adaptive_resolution_exponent = 0,
                    argument_blinks_per_second = None):
    ### animation_object: a static shape, list of shapes, scene, or dummyscene. 
    num_frames = int(num_frames)

    ### create current scene
    if animation_object is None:
        static_scene = Scene.make_empty()
    elif isinstance(animation_object, DummyScene) or isinstance(animation_object, Scene):
        static_scene = animation_object
    elif isinstance(animation_object, list):
        static_scene = Scene(animation_object)
    else:
        static_scene = Scene([animation_object])

    ### Create the List of Cameras to be Used
    if static_cam is None:
        static_cam = Camera.calibrated_AO_behind()
    
    ### Initialize background and foreground
    if background_scene is None:
        background_scene = Scene.make_empty()
   
    if previous_foreground_scene is None:
        previous_foreground_scene = Scene.make_empty()

    if previous_dummy_scene is None:
        previous_dummy_scene = DummyScene.make_empty()

    ### dim or lumify scenes based on whether they're argument or previous
    static_scene_brightened = static_scene.dim(argument_luminosity)
    static_scene_dimmed = static_scene.dim(1 / argument_luminosity) ### this passes into the next_animation_dictionary

    ### create joint_annotated_points -> previous points and current foreground point
    if annotated_points is None: annotated_points = []
    if previous_annotated_points is None: previous_annotated_points = []

    ### cull previous annotated points if they are included in annotated points (this can be useful if one wants to keep a text lit up)
    previous_annotated_points = [p for p in previous_annotated_points if p not in annotated_points]

    ### change text luminosity 
    argument_annotated_points_brightened = [p.dim(argument_luminosity) for p in annotated_points]
    argument_annotated_points_dimmed = [p.dim(1 / argument_luminosity) for p in annotated_points] ### this passes into the next_animation_dictionary
    annotated_points_to_show = [*argument_annotated_points_brightened, *previous_annotated_points]

    ### create previous and base views
    current_view = View.make(static_cam, static_scene_brightened, annotatedPoints = annotated_points_to_show, resolution = argument_resolution, adaptive_resolution_exponent = adaptive_resolution_exponent) 
    front_view = View.make(static_cam, previous_foreground_scene, resolution = foreground_resolution, adaptive_resolution_exponent = adaptive_resolution_exponent) 
    dummy_view = View.make(static_cam, previous_dummy_scene, resolution = dummy_resolution)
    base_view = View.make(static_cam, background_scene, resolution = background_resolution)

    if argument_blinks_per_second is None:
        joint_view = base_view.combine(dummy_view).combine(front_view).combine(current_view)
        static_frame = joint_view.snap()
        myFrms = BGRFrames.from_frame(static_frame, repetitions = num_frames)
    else:
        on_view = base_view.combine(dummy_view).combine(front_view).combine(current_view)
        off_view = base_view.combine(dummy_view).combine(front_view)
        onFrm = on_view.snap()
        offFrm = off_view.snap()
        
        myFrms = blinking(onFrm, offFrm, blinks_per_second = argument_blinks_per_second, duration_in_frames = num_frames)

    ### make a kwarg dictionary for the next animation
    if isinstance(static_scene, Scene):
        next_animation_dictionary = {'previous_foreground_scene': previous_foreground_scene.combine(static_scene_dimmed, keep_which = 'other'),
                                     'previous_dummy_scene': previous_dummy_scene,
                                     'previous_annotated_points': [*argument_annotated_points_dimmed, *previous_annotated_points]}
    elif isinstance(static_scene, DummyScene):
        next_animation_dictionary = {'previous_foreground_scene': previous_foreground_scene,
                                     'previous_dummy_scene': previous_dummy_scene.combine(static_scene, keep_which = 'other'),
                                     'previous_annotated_points': [*argument_annotated_points_dimmed, *previous_annotated_points]}
    

    return myFrms, next_animation_dictionary

def divide_line(someLn, num_divisions, reverse_direction = False, dashed = False):
    if dashed:
        num_divisions = (num_divisions * 2) + 1 ### * 2, since we simply take every second element,
                                                ### + 1, since we want the last element to be filled in!

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

    if not dashed:
        return itpl_lines
    else:
        return itpl_lines[::2] ### so simple :)

### Define our Kernel Density Estimator
def epanechnikov(posArr_nonnan, showArr, weightArr = None,
                 normalize_density = True, bandwidth = 30, 
                 zero_color = (0, 0, 0), one_color = const.BGR['intense_red'], 
                 mix_colors = True, mix_alpha = True):
    
    diffArr = showArr[np.newaxis] - posArr_nonnan[:, np.newaxis, np.newaxis]
    norm_diffArr = diffArr / bandwidth
    norm_devArr = (norm_diffArr ** 2).sum(axis = -1)
    
    kde_values_for_each_kernel = (1 - norm_devArr) * (norm_devArr < 1)

    if weightArr is not None:
        kde_values_for_each_kernel = kde_values_for_each_kernel * weightArr[:, np.newaxis, np.newaxis]

    kde_values = kde_values_for_each_kernel.mean(axis = 0)
    
    if normalize_density:
        kde_values = kde_values / kde_values.max()

    bgr = np.array([zero_color, one_color]) ### bgr for kde = 0 and 1 resp.
    alpha = np.array([[0], [1]]) ### alpha for kde = 0 and 1 resp.
    
    kde_brdcst = kde_values[:, :, np.newaxis]
    bgr_brdcst = bgr[:, np.newaxis, np.newaxis, :]
    alpha_brdcst = alpha[:, np.newaxis, np.newaxis, :]
    
    if mix_colors:
        bgrArr = (1 - kde_brdcst) * bgr_brdcst[0] + kde_brdcst * bgr_brdcst[1]
    else:
        bgrArr = np.ones_like(kde_brdcst) * bgr_brdcst[1]

    if mix_alpha:
        alphaArr = (1 - kde_brdcst) * alpha_brdcst[0] + kde_brdcst * alpha_brdcst[1]
    else:
        alphaArr = np.ones_like(kde_brdcst) * alpha_brdcst[1]

    return bgrArr, alphaArr

def meshgrid_region(showReg, resolution = 1):
    ### delineate the region to show
    try:
        x1, y1 = showReg.p1.coords
        x2, y2 = showReg.p4.coords
    except:
        x1, y1, _ = showReg.p1.coords
        x2, y2, _ = showReg.p4.coords
    xv, yv = np.meshgrid(np.arange(x1, x2, step = 1 / resolution), np.arange(y1, y2, step = 1 / resolution))
    showArr = np.stack([xv, yv], axis = -1)
    return showArr

def kde_heatmap(myShots, showReg = None,
                position_func = 'get_first_bounce_position',
                weight_func_or_arr = None,
                bandwidth = 30, normalize_density = True,
                resolution = 1, 
                mix_colors = True, mix_alpha = True,
                zero_color = (0, 0, 0), one_color = const.BGR['intense_red']):
    ### Kernel Density Heatmap with first bounce positions, pictured within a region

    ### Gather courtlines
    if showReg is None:
        showReg = CourtLines.make().get_court_region()
    
    ### Get first bounce position
    if isinstance(position_func, str):
        posArr = getattr(myShots, position_func)()
    else:
        posArr = np.array([position_func(aShot) for aShot in myShots], dtype = float)

    is_not_nan = ~np.isnan(posArr).all(axis = 1)
    posArr_nonnan = posArr[is_not_nan]
    
    ### figure out if weights need to be applied ###
    if weight_func_or_arr is None:
        weightArr = None
    elif isinstance(weight_func_or_arr, np.ndarray):
        weightArr = weight_func_or_arr[is_not_nan]
    elif isinstance(weight_func_or_arr, str):
        weightArr = getattr(myShots, weight_func_or_arr)()[is_not_nan]
    else:
        weightArr = np.array([weight_func_or_arr(aShot) for aShot in myShots], dtype = float)[is_not_nan]

    ### delineate the region to show
    showArr = meshgrid_region(showReg, resolution = resolution)
    num_x, num_y, _ = showArr.shape
    
    ### Prepare the resulting Scene
    bgrArr, alphaArr = epanechnikov(posArr_nonnan, showArr, weightArr = weightArr, 
                                    bandwidth = bandwidth, 
                                    zero_color = zero_color, one_color = one_color, 
                                    mix_colors = mix_colors, mix_alpha = mix_alpha,
                                    normalize_density = normalize_density)
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

def kde_hillmap(myShots, 
                showReg = None, resolution = 1,
                position_func = 'get_first_bounce_position',
                weight_func_or_arr = None,
                bandwidth = 30, max_height = 30, normalize_density = True,
                mix_colors = True, mix_alpha = True,
                zero_color = (0, 0, 0), one_color = const.BGR['intense_red'],
                threshold_proportion = None, epsilon = 0.005, threshold_color = const.BGR['intense_red']):
    
    ### Gather courtlines
    if showReg is None:
        showReg = CourtLines.make().get_court_region()
    
    ### Get first bounce position
    if isinstance(position_func, str):
        posArr = getattr(myShots, position_func)()
    else:
        posArr = np.array([position_func(aShot) for aShot in myShots], dtype = float)

    is_not_nan = ~np.isnan(posArr).all(axis = 1)
    posArr_nonnan = posArr[is_not_nan]
        
    ### figure out if weights need to be applied ###
    if weight_func_or_arr is None:
        weightArr = None
    elif isinstance(weight_func_or_arr, np.ndarray):
        weightArr = weight_func_or_arr[is_not_nan]
    elif isinstance(weight_func_or_arr, str):
        weightArr = getattr(myShots, weight_func_or_arr)()[is_not_nan]
    else:
        weightArr = np.array([weight_func_or_arr(aShot) for aShot in myShots], dtype = float)[is_not_nan]

    showArr = meshgrid_region(showReg, resolution = resolution)
    num_x, num_y, _ = showArr.shape
    
    ### Prepare the resulting Scene
    ### As opposed to above, we use the alpha channel to tell us about the height, here
    bgrArr, norm_heightArr = epanechnikov(posArr_nonnan, showArr, weightArr = weightArr,
                                          bandwidth = bandwidth, 
                                          zero_color = zero_color, one_color = one_color, 
                                          mix_colors = mix_colors, mix_alpha = True, ### mix alpha is not passed in to epanechnikov! used later.
                                          normalize_density = normalize_density)
    
    bgrArr_long = bgrArr.reshape((num_x * num_y, 3))
    
    ### Optionally -> mark out a threshold
    if threshold_proportion is not None:
        sortArr = np.sort(norm_heightArr.flatten())
        threshArr = sortArr.cumsum() / norm_heightArr.sum() > threshold_proportion
        threshold_density = sortArr[threshArr.argmax()]
        
        density_is_above = threshold_density - epsilon < norm_heightArr
        density_is_below = norm_heightArr < threshold_density + epsilon
        
        long_density_filter = (density_is_above & density_is_below).reshape((num_x * num_y))

        bgrArr_long_replace = np.tile(threshold_color, reps = (num_x * num_y, 1))
        bgrArr_long = np.where(long_density_filter[:, np.newaxis], bgrArr_long_replace, bgrArr_long)

    heightArr = norm_heightArr * max_height
    heightArr_long = heightArr.reshape((num_x * num_y, 1))

    showArr_long = showArr.reshape((num_x * num_y, 2))
    showArr_long = np.concatenate([showArr_long, heightArr_long], axis = 1)

    alphaArr_long = np.ones((num_x * num_y, 1))
    if mix_alpha:
        alphaArr_long = alphaArr_long * heightArr_long / heightArr_long.max()

    kdeScn = DummyScene([showArr_long], [bgrArr_long], [alphaArr_long])
    return kdeScn

def kde_contourmap(myShots, 
                   showReg = None, resolution = 1, 
                   position_func = 'get_first_bounce_position',
                   weight_func_or_arr = None,
                   bandwidth = 30, threshold_proportion = 0.5, 
                   epsilon = 0.005, n_components = None,
                   fill = False, max_height = 30, display_at_height = True,
                   normalize_density = True,
                   preserve_showarr_shape = True, 
                   contour_color = const.BGR['intense_red'],
                   contour_alpha = 1):
    
    ### Gather courtlines
    if showReg is None:
        showReg = CourtLines.make().get_court_region()
    
    ### Get first bounce position
    if isinstance(position_func, str):
        posArr = getattr(myShots, position_func)()
    else:
        posArr = np.array([position_func(aShot) for aShot in myShots], dtype = float)

    is_not_nan = ~np.isnan(posArr).all(axis = 1)
    posArr_nonnan = posArr[is_not_nan]
    
    ### figure out if weights need to be applied ###
    if weight_func_or_arr is None:
        weightArr = None
    elif isinstance(weight_func_or_arr, np.ndarray):
        weightArr = weight_func_or_arr[is_not_nan]
    elif isinstance(weight_func_or_arr, str):
        weightArr = getattr(myShots, weight_func_or_arr)()[is_not_nan]
    else:
        weightArr = np.array([weight_func_or_arr(aShot) for aShot in myShots], dtype = float)[is_not_nan]

    showArr = meshgrid_region(showReg, resolution = resolution)
    num_x, num_y, num_coords = showArr.shape
    
    ### Prepare the resulting Scene
    ### Here, we simply get out the raw kde values by ignoring the BGR channel and turning off normalization
    _, kdeArr = epanechnikov(posArr_nonnan, showArr, weightArr = weightArr,
                             bandwidth = bandwidth, 
                             normalize_density = normalize_density, mix_alpha = True)
    kdeArr = kdeArr[:, :, 0]

    ### find threshold density
    sortArr = np.sort(kdeArr.flatten())
    threshArr = sortArr.cumsum() / kdeArr.sum() > threshold_proportion
    threshold_density = sortArr[threshArr.argmax()]
    
    density_is_above = threshold_density - epsilon < kdeArr
    density_is_below = kdeArr < threshold_density + epsilon

    ### cluster hills, so we can group by cluster and understand the contained probability mass
    if n_components is not None:
        exceedArr = showArr[density_is_above]
         
        from sklearn.mixture import BayesianGaussianMixture
        clustering = BayesianGaussianMixture(n_components = n_components, covariance_type = 'full').fit(exceedArr)
        
        cluster_data = []
        cluster_idxs, cluster_counts = np.unique(clustering.predict(exceedArr), return_counts = True)

        for idx in cluster_idxs:
            cluster_shares = cluster_counts / cluster_counts.sum()
            clusterDict = {'cluster_centroid': tuple(clustering.means_[idx]),
                           'cluster_share': cluster_shares[idx]}
            cluster_data.append(clusterDict)
    else:
        cluster_data = None

    if display_at_height:
        heightArr = np.full((num_x, num_y, 1), fill_value = threshold_density * max_height)
    else:
        heightArr = np.full((num_x, num_y, 1), fill_value = 0)
    showArr_with_height = np.concatenate([showArr, heightArr], axis = 2)

    show_condition = density_is_above & density_is_below if not fill else density_is_above
    if preserve_showarr_shape:
        showArr_long = np.where(show_condition[:, :, np.newaxis], showArr_with_height, np.nan).reshape((num_x * num_y, 3))
    else:
        showArr_long = showArr_with_height[show_condition]

    bgrArr_long = np.ones((num_x * num_y, 3)) * contour_color
    alphaArr_long = np.ones((num_x * num_y, 1)) * contour_alpha

    kdeScn = DummyScene([showArr_long], [bgrArr_long], [alphaArr_long])

    return kdeScn, cluster_data


def scatter_markmap(myShots, showReg = None,
                    position_func = 'get_first_bounce_position',
                    color_func = None,
                    normalize = True,
                    mix_colors = True, mix_alpha = False,
                    flat_along_dimension = 'z',
                    zero_color = (0, 0, 0), one_color = const.BGR['intense_red']):        
    from ATMANpy.types.ShapesTennis3D import Mark
    from ATMANpy.types.Shapes3D import Point3D
    from ATMANpy.types.Points import Point2D

    ### Get first bounce position (or different position function)
    if isinstance(position_func, str):
        posArr = getattr(myShots, position_func)()
    else:
        posArr = np.array([position_func(aShot) for aShot in myShots], dtype = float)
    
    position_is_not_none = ~np.isnan(posArr).all(axis = 1)
    posArr_nonnan = posArr[position_is_not_none]
    
    ### Check for dimensionality, and make consistently 3D
    frames, dim = posArr_nonnan.shape
    if dim == 2:
        if flat_along_dimension == 'x':
            posArr_nonnan = np.insert(posArr_nonnan, 0, 0, axis = 1)
        elif flat_along_dimension == 'y':
            posArr_nonnan = np.insert(posArr_nonnan, 1, 0, axis = 1)
        elif flat_along_dimension == 'z':
            posArr_nonnan = np.insert(posArr_nonnan, 2, 0, axis = 1)

    ### Get color function (if it is not None)
    if color_func is not None:
        if isinstance(color_func, str):
            colArr = getattr(myShots, color_func)()
        else:
            colArr = np.array([color_func(aShot) for aShot in myShots], dtype = float)
    else:
        colArr = np.ones_like(posArr)

    colArr_nonnan = colArr[position_is_not_none]

    if normalize:
        colArr_nonnan = colArr_nonnan / colArr_nonnan.max()

    shapeList = []
    for coords, color in zip(posArr_nonnan, colArr_nonnan):
        
        if showReg is not None:
            ### make a point from coords, and check if it is eligible
            if flat_along_dimension == 'x':
                coords2D = coords[1], coords[2]
            elif flat_along_dimension == 'y':
                coords2D = coords[0], coords[2]
            elif flat_along_dimension == 'z':
                coords2D = coords[0], coords[1]

            pt2D = Point2D(coords2D)
            if not pt2D.within(showReg):
                continue

        ### figure out color mixing
        if mix_colors:
            mixed_tuple = tuple([hi * color + lo * (1 - color) for hi, lo in zip(one_color, zero_color)])
        else:
            mixed_tuple = one_color
        
        ### figure out alpha mixing
        if mix_alpha:
            mixed_alpha = color ### note that color is basically a lambda (see its use above), so we can pass it directly to alpha
        else:
            mixed_alpha = 1

        ### make a mark
        pt3D = Point3D(coords)
        myMark = Mark.make(pt3D, color = mixed_tuple, alpha = mixed_alpha, flat_along_dimension = flat_along_dimension)
        shapeList.append(myMark)

    return Scene(shapeList)

def cluster_data_to_points(cluster_data, text_func = None, thickness = 1, size = 1, color = const.BGR['intense_red'],
                           up = 0, down = 0, front = 0, back = 0, left = 0, right = 0):
    if text_func is None:
        text_func = lambda share: f'{share * 100:.0f}%'

    from ATMANpy.types.Shapes3D import AnnotatedPoint3D
    clusterPts = []
    for data in cluster_data:
        x, y = data['cluster_centroid']
        share = data['cluster_share']

        text = text_func(share)

        myPnt = AnnotatedPoint3D((x, y, 0), color = color, text = text, 
                                 thickness = thickness, size = size)
        myPnt = myPnt.up(up).down(down).front(front).back(back).left(left).right(right)
        clusterPts.append(myPnt)

    return clusterPts

def players(myShots, 
            hit_player_color = const.BGR['smokey_grey'], rec_player_color = const.BGR['smokey_grey']):
    from ATMANpy.types.ShapesTennis3D import Player, Racquet

    ### PREP PLAYERS
    hit_x, hit_y = myShots.get_player_hit_position().mean(axis = 0)
    rec_x, rec_y = myShots.get_player_rec_position().mean(axis = 0)

    hitPly = Player.make((hit_x, hit_y), color = hit_player_color)
    recPly = Player.make((rec_x, rec_y), color = rec_player_color)

    plyScn = Scene([hitPly, recPly])
    return plyScn
    
def players_and_racquets(myShots, 
                        hit_player_color = const.BGR['smokey_grey'], rec_player_color = const.BGR['smokey_grey'],
                        hit_player_distance = 10, rec_player_distance = 10,
                        hit_player_dir_vert = 0, rec_player_dir_vert = 0):
    from ATMANpy.types.ShapesTennis3D import Player, Racquet

    ### PREP PLAYERS
    hit_x, hit_y = myShots.get_player_hit_position().mean(axis = 0)
    rec_x, rec_y = myShots.get_player_rec_position().mean(axis = 0)

    hitPly = Player.make((hit_x, hit_y), color = hit_player_color)
    recPly = Player.make((rec_x, rec_y), color = rec_player_color)

    hitRac = Racquet.make_at_player(hitPly, color = hit_player_color,
                                    distance_from_player = hit_player_distance, dir_vert = hit_player_dir_vert)
    recRac = Racquet.make_at_player(recPly, color = rec_player_color,
                                    distance_from_player = rec_player_distance, dir_vert = rec_player_dir_vert)
    
    plyScn = Scene([hitPly, hitRac, recPly, recRac])
    return plyScn
    
def gmm_empirical_density_rings(myShots, threshold_proportion = 0.5, n_components = 5):
    from ATMANpy.types.Shapes3D import Circle3D, Circles3D

    bounceArr = myShots.get_first_bounce_position().astype('float')
    bounceArr = bounceArr[~np.isnan(bounceArr).any(axis = 1), :2]

    from sklearn import mixture
    gmm = mixture.BayesianGaussianMixture(n_components = n_components, covariance_type = 'full')
    clustering = gmm.fit(bounceArr)
    
    ### theoretically, we should be able to use the means and covariances to calculate two parsimonious balls within which
    ### there lie threshold_proportion of observations. 
    ### However, a faster (and cruder) way of finding these (for a spherical GMM) is 
        # taking the threshold_proportion highest likelihood samples,
        # grouping them by the mixture they belong to.
        # grabbing the lowest likelihood sample for each of these mixtures
        # calculating its distance from the mixture mean.
        # equivalently, we can simply grab the largest distances for each of these mixtures (since the covariance is spherical)
    likelihoods = clustering.score_samples(bounceArr)        
    predictions = clustering.predict(bounceArr)

    threshold_likelihood = np.quantile(likelihoods, 1 - threshold_proportion)

    mixture_circles = []
    mixture_shares = []
    for mixture_index in np.unique(predictions):
        mixture_filter = (predictions == mixture_index) & (likelihoods > threshold_likelihood)
        if mixture_filter.sum() == 0:
            continue
        mixture_mean = clustering.means_[mixture_index]

        mixture_distances = ((bounceArr[mixture_filter] - mixture_mean) ** 2).sum(axis = 1) ** 0.5
        radius = np.max(mixture_distances)

        mixCrc = Circle3D(tuple([*mixture_mean, 0]), radius = radius)
        mixture_circles.append(mixCrc)
        mixture_shares.append(mixture_filter.mean())

    return Circles3D(mixture_circles), mixture_shares

def kill_nans_in_vectorized_line(aLn):
    leftPt, rightPt = aLn.leftPt, aLn.rightPt

    lFilt = np.isnan(leftPt.coords).all(axis = 0)
    rFilt = np.isnan(rightPt.coords).all(axis = 0)
    
    ### since nans typically occur in later frames, lFilt <= rFilt ...
    assert lFilt.sum() <= rFilt.sum() 
    ### ... therefore, we use rFilt

    l_coords = leftPt.coords[:, ~rFilt]
    r_coords = rightPt.coords[:, ~rFilt]

    return Line3D(Point3D(l_coords), Point3D(r_coords), color = aLn.color, alpha = aLn.alpha, thickness = aLn.thickness)

def ball_flight(myShots, line_interpolation_factor = None):
    splineSF = myShots.feat_spline_simul(color = const.BGR['ball_yellow'], which_endpoint = 'actual')
    shapes = list(splineSF.values())
    shapes_vectorized = [f.vectorize() for f in shapes if f is not None]
    shapes_vectorized_nanfilt = [kill_nans_in_vectorized_line(aLn) for aLn in shapes_vectorized]

    if line_interpolation_factor is None:
        shapes_vectorized_nanfilt_expanded = shapes_vectorized_nanfilt
    else:
        shapes_vectorized_nanfilt_expanded = [subLn for aLn in shapes_vectorized_nanfilt for subLn in divide_line(aLn, num_divisions = line_interpolation_factor)]
    
    return shapes_vectorized_nanfilt_expanded