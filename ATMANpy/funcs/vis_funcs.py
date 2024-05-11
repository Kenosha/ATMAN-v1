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

def interpolate_dummy_scenes(start_dummyScn, end_dummyScn, num_frames):
    fs_len, ls_len = len(start_dummyScn.rastArrs), len(end_dummyScn.rastArrs)
    assert fs_len == ls_len

    scenes = []
    for frame in range(num_frames):
        lmbd = frame / num_frames
        
        itpl_rastArrs = []
        itpl_colArrs = []
        itpl_alphaArrs = []
        for idx in range(fs_len):
            fs_rastArr, ls_rastArr = start_dummyScn.rastArrs[idx], end_dummyScn.rastArrs[idx]
            fs_colArr, ls_colArr = start_dummyScn.colArrs[idx], end_dummyScn.colArrs[idx]
            fs_alphaArr, ls_alphaArr = start_dummyScn.alphaArrs[idx], end_dummyScn.alphaArrs[idx]

            itpl_rastArr = (1 - lmbd) * fs_rastArr + lmbd * ls_rastArr
            itpl_colArr = (1 - lmbd) * fs_colArr + lmbd * ls_colArr
            itpl_alphaArr = (1 - lmbd) * fs_alphaArr + lmbd * ls_alphaArr

            itpl_rastArrs.append(itpl_rastArr)
            itpl_colArrs.append(itpl_colArr)
            itpl_alphaArrs.append(itpl_alphaArr)

        frameScn = DummyScene(itpl_rastArrs, itpl_colArrs, itpl_alphaArrs)
        scenes.append(frameScn)

    return scenes

def animate_sequence(animation_sequence_objects, 
                    background_scene = None, previous_foreground_scene = None, previous_dummy_scene = None,
                    argument_resolution = 1, background_resolution = 1, foreground_resolution = 1, dummy_resolution = 1, 
                    start_cam = None, stop_cam = None,
                    cumulative_foreground_transparence = None, 
                    annotatedPoints = None,
            text_lines = None, text_size = 1, text_color = const.BGR['strong_pink'], text_where = (50, 50), text_offset = 50):
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
        previous_dummy_scene = Scene.make_empty()
    
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
        current_view = View.make(current_cam, current_scene, annotatedPoints = annotatedPoints, resolution = argument_resolution) 
    
        ### create previous and base views
        front_view = View.make(current_cam, previous_foreground_scene, resolution = foreground_resolution)
        dummy_view = View.make(current_cam, previous_dummy_scene, resolution = dummy_resolution)
        base_view = View.make(current_cam, background_scene, resolution = background_resolution)

        joint_view = base_view.combine(dummy_view).combine(front_view).combine(current_view)
        
        myFrm = joint_view.snap()

        if text_lines is not None:
            pos_x, pos_y = text_where
            for textStr in text_lines:
                myFrm = myFrm.put(textStr, where = (pos_x, pos_y), color = text_color, size = text_size)
                pos_y += text_offset

        frameList.append(myFrm)

    myFrms = BGRFrames(np.array([f.array for f in frameList]))

    ### make a kwarg dictionary for the next animation
    if isinstance(current_scene, Scene):
        next_animation_dictionary = {'previous_foreground_scene': previous_foreground_scene.combine(current_scene),
                                     'previous_dummy_scene': previous_dummy_scene}
    elif isinstance(current_scene, DummyScene):
        next_animation_dictionary = {'previous_foreground_scene': previous_foreground_scene,
                                     'previous_dummy_scene': previous_dummy_scene.combine(current_scene)}

    return myFrms, next_animation_dictionary


def animate_cam_motion(animation_object, num_frames = None,
                        background_scene = None, previous_foreground_scene = None, previous_dummy_scene = None, ### these are static/previous scenes that are not animated, but can be faded out/in
                        argument_resolution = 1, background_resolution = 1, foreground_resolution = 1, dummy_resolution = 1, 
                        start_cam = None, stop_cam = None,
                        cumulative_foreground_transparence = 0, 
                        annotatedPoints = None,
                        text_lines = None, text_size = 1, text_color = const.BGR['strong_pink'], text_where = (50, 50), text_offset = 50):
    ### animation_object: a static shape, list of shapes, scene, or dummyscene. 

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
        previous_dummy_scene = Scene.make_empty()
    
    cams = interpolate_camera(start_cam, stop_cam, num_frames) 

    frameList = []
    for idx in range(num_frames):
        current_cam = cams[idx]

        ### fade previous foreground scenes
        if cumulative_foreground_transparence is not None:
            previous_foreground_scene = previous_foreground_scene.transparify(cumulative_foreground_transparence)
            previous_dummy_scene = previous_dummy_scene.transparify(cumulative_foreground_transparence)
        
        current_view = View.make(current_cam, static_scene, annotatedPoints = annotatedPoints, resolution = argument_resolution) 
    
        ### create previous and base views
        front_view = View.make(current_cam, previous_foreground_scene, resolution = foreground_resolution)
        dummy_view = View.make(current_cam, previous_dummy_scene, resolution = dummy_resolution)
        base_view = View.make(current_cam, background_scene, resolution = background_resolution)

        joint_view = base_view.combine(dummy_view).combine(front_view).combine(current_view)
        
        myFrm = joint_view.snap()

        if text_lines is not None:
            pos_x, pos_y = text_where
            for textStr in text_lines:
                myFrm = myFrm.put(textStr, where = (pos_x, pos_y), color = text_color, size = text_size)
                pos_y += text_offset

        frameList.append(myFrm)
        if text_lines is not None:
            pos_x, pos_y = text_where
            for textStr in text_lines:
                myFrm = myFrm.put(textStr, where = (pos_x, pos_y), color = text_color, size = text_size)
                pos_y += text_offset

        frameList.append(myFrm)

    myFrms = BGRFrames(np.array([f.array for f in frameList]))

    ### make a kwarg dictionary for the next animation
    if isinstance(static_scene, Scene):
        next_animation_dictionary = {'previous_foreground_scene': previous_foreground_scene.combine(static_scene),
                                     'previous_dummy_scene': previous_dummy_scene}
    elif isinstance(static_scene, DummyScene):
        next_animation_dictionary = {'previous_foreground_scene': previous_foreground_scene,
                                     'previous_dummy_scene': previous_dummy_scene.combine(static_scene)}

    return myFrms, next_animation_dictionary


def animate_static(animation_object, num_frames = 100, static_cam = None, 
                    background_scene = None, previous_foreground_scene = None, previous_dummy_scene = None,
                    argument_resolution = 1, background_resolution = 1, foreground_resolution = 1, dummy_resolution = 1, 
                    annotatedPoints = None,
                    text_lines = None, text_size = 1, text_color = const.BGR['strong_pink'], text_where = (50, 50), text_offset = 50):
    ### animation_object: a static shape, list of shapes, scene, or dummyscene. 

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
        previous_dummy_scene = Scene.make_empty()
    
    ### create previous and base views
    current_view = View.make(static_cam, static_scene, annotatedPoints = annotatedPoints, resolution = argument_resolution) 
    front_view = View.make(static_cam, previous_foreground_scene, resolution = foreground_resolution)
    dummy_view = View.make(static_cam, previous_dummy_scene, resolution = dummy_resolution)
    base_view = View.make(static_cam, background_scene, resolution = background_resolution)

    joint_view = base_view.combine(dummy_view).combine(front_view).combine(current_view)
    static_frame = joint_view.snap()

    ### MANAGE TEXT
    if text_lines is not None:
        pos_x, pos_y = text_where
        for textStr in text_lines:
            static_frame = static_frame.put(textStr, where = (pos_x, pos_y), color = text_color, size = text_size)
            pos_y += text_offset

    ### COLLATE STATIC FRAMES 
    frameList = [static_frame for _ in range(num_frames)]
    myFrms = BGRFrames(np.array([f.array for f in frameList]))
    
    ### make a kwarg dictionary for the next animation
    if isinstance(static_scene, Scene):
        next_animation_dictionary = {'previous_foreground_scene': previous_foreground_scene.combine(static_scene),
                                     'previous_dummy_scene': previous_dummy_scene}
    elif isinstance(static_scene, DummyScene):
        next_animation_dictionary = {'previous_foreground_scene': previous_foreground_scene,
                                     'previous_dummy_scene': previous_dummy_scene.combine(static_scene)}

    return myFrms, next_animation_dictionary

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

### Define our Kernel Density Estimator
def epanechnikov(posArr_nonnan, showArr, normalize_density = True, bandwidth = 30, 
                 zero_color = (0, 0, 0), one_color = const.BGR['intense_red'], 
                 mix_colors = True, mix_alpha = True):
    
    diffArr = showArr[np.newaxis] - posArr_nonnan[:, np.newaxis, np.newaxis]
    norm_diffArr = diffArr / bandwidth
    norm_devArr = (norm_diffArr ** 2).sum(axis = -1)
    
    kde_values_for_each_kernel = (1 - norm_devArr) * (norm_devArr < 1)
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
    x1, y1 = showReg.p1.coords
    x2, y2 = showReg.p4.coords
    xv, yv = np.meshgrid(np.arange(x1, x2, step = 1 / resolution), np.arange(y1, y2, step = 1 / resolution))
    showArr = np.stack([xv, yv], axis = -1)
    return showArr

def kde_heatmap(myShots, showReg = None,
                position_func = 'get_first_bounce_position',
                bandwidth = 30,
                resolution = 1, 
                mix_colors = True, mix_alpha = True,
                zero_color = (0, 0, 0), one_color = const.BGR['intense_red']):
    ### Kernel Density Heatmap with first bounce positions, pictured within a region

    ### Gather courtlines
    if showReg is None:
        showReg = CourtLines.make().get_court_region()
    
    ### Get first bounce position
    coords = getattr(myShots, position_func)()
    posArr = np.array(coords, dtype = 'float32')[:, :2]
    posArr_nonnan = posArr[~np.isnan(posArr).all(axis = 1)]
    
    ### delineate the region to show
    showArr = meshgrid_region(showReg, resolution = resolution)
    num_x, num_y, _ = showArr.shape
    
    ### Prepare the resulting Scene
    bgrArr, alphaArr = epanechnikov(posArr_nonnan, showArr, bandwidth = bandwidth, 
                                    zero_color = zero_color, one_color = one_color, 
                                    mix_colors = mix_colors, mix_alpha = mix_alpha)
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
                bandwidth = 30, max_height = 30,
                mix_colors = True, 
                zero_color = (0, 0, 0), one_color = const.BGR['intense_red']):
    
    ### Gather courtlines
    if showReg is None:
        showReg = CourtLines.make().get_court_region()
    
    ### Get first bounce position
    posArr = getattr(myShots, position_func)()
    posArr_nonnan = posArr[~np.isnan(posArr).all(axis = 1)]
    
    showArr = meshgrid_region(showReg, resolution = resolution)
    num_x, num_y, _ = showArr.shape
    
    ### Prepare the resulting Scene
    ### As opposed to above, we use the alpha channel to tell us about the height, here
    bgrArr, norm_heightArr = epanechnikov(posArr_nonnan, showArr, bandwidth = bandwidth, 
                                    zero_color = zero_color, one_color = one_color, 
                                    mix_colors = mix_colors, mix_alpha = True)
    
    bgrArr_long = bgrArr.reshape((num_x * num_y, 3))
    
    heightArr = norm_heightArr * max_height
    heightArr_long = heightArr.reshape((num_x * num_y, 1))

    showArr_long = showArr.reshape((num_x * num_y, 2))
    showArr_long = np.concatenate([showArr_long, heightArr_long], axis = 1)

    alphaArr_long = np.ones((num_x * num_y, 1))

    kdeScn = DummyScene([showArr_long], [bgrArr_long], [alphaArr_long])
    return kdeScn

def kde_contourmap(myShots, 
                   showReg = None, resolution = 1, 
                   position_func = 'get_first_bounce_position',
                   bandwidth = 30, threshold_proportion = 0.5, 
                   epsilon = 0.005, n_components = None,
                   fill = False, preserve_showarr_shape = True,
                   contour_color = const.BGR['intense_red'],
                   contour_alpha = 1):
    
    ### Gather courtlines
    if showReg is None:
        showReg = CourtLines.make().get_court_region()
    
    ### Get first bounce position
    coords = getattr(myShots, position_func)()
    posArr = np.array(coords, dtype = 'float32')[:, :2]
    posArr_nonnan = posArr[~np.isnan(posArr).all(axis = 1)]
    
    showArr = meshgrid_region(showReg, resolution = resolution)
    num_x, num_y, num_coords = showArr.shape
    
    ### Prepare the resulting Scene
    ### Here, we simply get out the raw kde values by ignoring the BGR channel and turning off normalization
    _, kdeArr = epanechnikov(posArr_nonnan, showArr, bandwidth = bandwidth, 
                            normalize_density = False, mix_alpha = True)
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
    
    show_condition = density_is_above & density_is_below if not fill else density_is_above
    if preserve_showarr_shape:
        showArr_long = np.where(show_condition[:, :, np.newaxis], showArr, np.nan).reshape((num_x * num_y, num_coords))
    else:
        showArr_long = showArr[show_condition]

    showArr_long = np.concatenate([showArr_long, np.zeros((num_x * num_y, 1))], axis = 1)

    bgrArr_long = np.ones((num_x * num_y, 3)) * contour_color
    alphaArr_long = np.ones((num_x * num_y, 1)) * contour_alpha

    kdeScn = DummyScene([showArr_long], [bgrArr_long], [alphaArr_long])

    return kdeScn, cluster_data

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