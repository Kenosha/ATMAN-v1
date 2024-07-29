
import numpy as np

from ATMANpy.types.Shapes3D import Region3D, Regions3D, AnnotatedPoint3D
from ATMANpy.types.Visual3D import DummyScene, Scene, Camera, View

from ATMANpy.types.Frames import Frame
from ATMANpy.types.MultiFrames import BGRFrames

import ATMANpy.auxil.constants as const

from dataclasses import dataclass

def make_region_in(subReg, offset = 0.1, color = const.BGR['smokey_grey']):
    assert offset < 0.5
    myReg = subReg.rescale(1 - offset * 2)
    myReg = myReg.recolor(color)
    return myReg

def make_text_in(subReg, textStr, size = 1, thickness = 1, color = const.BGR['leafy_green'], offset = 0):
    textPt = subReg['Left'].center_point().annotate(textStr)
    textPt = textPt.resize(size).rethicken(thickness).recolor(color).right(offset)
    return textPt

@dataclass(frozen = True)
class Legend:
    element_parameters: list
    legend_dimensions: tuple = (600, 400)
    render_height: int = 100
    resolution: int = 10
    split_denominator: int = 4
    borders: bool = True
    header_text: str = 'Legend'
    header_size: int = 2
    header_thickness: int = 2
    header_offset: int = 10
    header_color: tuple = const.BGR['smokey_grey']

    @staticmethod
    def make_empty(legend_dimensions = (600, 400), render_height = 100, resolution = 10, split_denominator = 4, borders = True,
                   header_text = 'Legend', header_size = 2, header_thickness = 2, header_offset = 10, header_color = const.BGR['smokey_grey']):
        return Legend([], 
                      legend_dimensions = legend_dimensions, 
                      render_height = render_height,
                      resolution = resolution,
                      split_denominator = split_denominator,
                      borders = borders,
                      header_text = header_text,
                      header_size = header_size,
                      header_thickness = header_thickness,
                      header_offset = header_offset,
                      header_color = header_color)
    
    def add(self, parameter_dict):
        return Legend([*self.element_parameters, parameter_dict], 
                      legend_dimensions = self.legend_dimensions, 
                      render_height = self.render_height,
                      resolution = self.resolution,
                      borders = self.borders,
                      split_denominator = self.split_denominator,
                      header_text = self.header_text,
                      header_size = self.header_size,
                      header_thickness = self.header_thickness,
                      header_offset = self.header_offset,
                      header_color = self.header_color)
    
    def render(self):
        legCam = Camera.birds_eye(self.render_height, sensor_dims = self.legend_dimensions)
        legReg = legCam.approximate_fov(self.render_height)

        legLines = legReg.lineList
        legLines = legLines.resize(0.25)

        if not self.borders:
            legLines = legLines.realpha(0)

        num_els = len(self.element_parameters)

        subRegs = legReg.subdivide(elements = num_els + 1, how = 'horizontal')
        headerReg, otherRegs = subRegs[0], subRegs[1:]

        headerPt = make_text_in(headerReg, self.header_text, 
                                size = self.header_size, 
                                thickness = self.header_thickness,
                                offset = self.header_offset,
                                color = self.header_color)

        legend_shapes, annotated_points = [legLines], [headerPt]
        for params, subReg in zip(self.element_parameters, otherRegs):
            
            if params is None: ### this is so we can have a legend of fixed subdivisions with empty elements -> None as a placeholder
                continue

            subReg_left, subReg_right = subReg.split(numerator = 1, denominator = self.split_denominator, how = 'vertical')

            ### shape on the left side of the legend element
            try:
                shape_color = params['shape_color']
                shape_offset = params['shape_offset']
            except KeyError:
                pass
            else:
                shapeReg = make_region_in(subReg_left, offset = shape_offset, color = shape_color)
                legend_shapes.append(shapeReg)

            ### text to add on the left side of the legend element, on top of shape potentially
            try:
                text_string_shape = params['text_string_shape']
                text_color_shape = params['text_color_shape']
                text_size_shape = params['text_size_shape']
                text_thickness_shape = params['text_thickness_shape']
                text_offset_shape = params['text_offset_shape']
            except KeyError:
                pass
            else:
                textPt_left = make_text_in(subReg_left, text_string_shape, 
                                           size = text_size_shape, thickness = text_thickness_shape, 
                                           color = text_color_shape, offset = text_offset_shape)
                annotated_points.append(textPt_left)

            ### text on the right side of the legend element
            try:
                text_string = params['text_string']
                text_color = params['text_color']
                text_size = params['text_size']
                text_thickness = params['text_thickness']
                text_offset = params['text_offset']
            except KeyError:
                pass
            else:
                textPt_right = make_text_in(subReg_right, text_string, 
                                            size = text_size, thickness = text_thickness, 
                                            color = text_color, offset = text_offset)
                annotated_points.append(textPt_right)

        legScn = Scene(legend_shapes)
        legView = View.make(legCam, legScn, annotatedPoints = annotated_points, resolution = self.resolution)
        legFrm = legView.snap()

        return legFrm

def overlay(baseFrm, legFrm, top_left_point = None):
    from ATMANpy.types.Frames import BGRFrame
    from ATMANpy.types.MultiFrames import BGRFrames
    
    if top_left_point is None:
        tl_col, tl_row = 0, 0
    else:
        tl_col, tl_row = top_left_point.coords

    if isinstance(baseFrm, BGRFrame):
        bs_row, bs_col, _ = baseFrm.array.shape
        bs_t = None
    elif isinstance(baseFrm, BGRFrames):
        bs_t, bs_row, bs_col, _ = baseFrm.array.shape
    
    if isinstance(legFrm, BGRFrame):
        lg_row, lg_col, _ = legFrm.array.shape
        lg_t = None
    elif isinstance(legFrm, BGRFrames):
        lg_t, lg_row, lg_col, _ = legFrm.array.shape
    
    if lg_t is not None: ### if we have a legend with a time dimension...
        assert bs_t == lg_t

    assert tl_col + lg_col < bs_col and tl_row + lg_row < bs_row

    combArr = baseFrm.array.copy()
    legArr = legFrm.array.copy()

    ### bring to have four dims, so we can add in, and squeeze later
    if len(combArr.shape) == 3:
        combArr = combArr[np.newaxis]
    if len(legArr.shape) == 3:
        legArr = legArr[np.newaxis]

    combArr[:, tl_row:tl_row + lg_row, tl_col:tl_col + lg_col] = legArr

    if combArr.shape[0] == 1:    
        return BGRFrame(combArr[0])
    else:
        return BGRFrames(combArr)

class FramesRaster:
    ### Create rastered MultiFrames from subframes.
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        self.raster = np.full((rows, cols), fill_value = None).tolist()
        self.length = 0
        self.element_dims = None
        self.output_dims = None

    def add(self, someFrms, row, col,
            set_size = False): ### added element overwrites the size
        self.raster[row][col] = someFrms

        if isinstance(someFrms, Frame):
            BGRFrames.from_frame(someFrms, 1)

        if someFrms.length() > self.length or set_size:
            self.length = someFrms.length()
        
        dims = someFrms.cols(), someFrms.rows() ### note the switch here.
        if self.element_dims is not None:
            assert dims == self.element_dims
        else:
            self.element_dims = dims

        if self.output_dims is None:
            el_cols, el_rows = self.element_dims
            self.output_dims = el_cols * self.cols, el_rows * self.rows
        
    def render(self):
        ### idea: get all the element MultiFrames to have the target dimensions and the target length
        ### for None elements, create black 
        ### then, enlarge them, with padding corresponding to where they are placed in the raster
        ### then, simply add these new MultiFrames together.
        el_cols, el_rows = self.element_dims
        al_cols, al_rows = self.output_dims

        jointArr = np.zeros((self.length, al_rows, al_cols, 3), dtype = 'uint8')

        for row_idx, rows in enumerate(self.raster):
            for col_idx, someFrms in enumerate(rows):
                if someFrms is None:
                    continue
                
                addArr = someFrms.array.copy()
                
                ### if necessary, bring to proper length by repeating the last frame
                add_len = addArr.shape[0]
                if add_len < self.length:
                    lsArr = addArr[-1]
                    last_frames_to_add = self.length - add_len
                    repeatedFrms = np.stack([lsArr] * last_frames_to_add, axis = 0)
                    addArr = np.concatenate([addArr, repeatedFrms], axis = 0)

                addArr = addArr[:self.length]

                ### find right spot in joint array
                row_left, row_right = row_idx * el_rows, (row_idx + 1) * el_rows
                col_left, col_right = col_idx * el_cols, (col_idx + 1) * el_cols
                jointArr[:, row_left:row_right, col_left:col_right] = addArr

        return BGRFrames(jointArr)
