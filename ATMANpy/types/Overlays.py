
import numpy as np

from ATMANpy.types.Shapes3D import Region3D, Regions3D, AnnotatedPoint3D, Line3D, Lines3D
from ATMANpy.types.Visual3D import DummyScene, Scene, Camera, View

from ATMANpy.types.Frames import Frame
from ATMANpy.types.MultiFrames import BGRFrames

import ATMANpy.auxil.constants as const

from dataclasses import dataclass

from fractions import Fraction

def maybe_data_to_text(data_or_string, number_format):
    if isinstance(data_or_string, str):
        return data_or_string
    else:
        return f'{data_or_string:{number_format}}'

def make_region_in(subReg, offset = 0.1, color = const.BGR['smokey_grey']):
    assert offset < 0.5
    myReg = subReg.rescale(1 - offset * 2)
    myReg = myReg.recolor(color)
    return myReg

def make_text_in(subReg, data_or_string, 
                 size = 1, thickness = 1, color = const.BGR['leafy_green'], 
                 offset = 0, skip_after_how_many_symbols = None,
                 number_format = '.2f'):
    textStr = maybe_data_to_text(data_or_string, number_format)

    ### make word splits
    if skip_after_how_many_symbols is not None:
        words = textStr.split(' ')
        cumsums = np.cumsum([len(word) for word in words])
        word_in_which_line = cumsums // skip_after_how_many_symbols

        word_splits = []
        current_split = []
        current_line = 0
        for word, which in zip(words, word_in_which_line):
            if which == current_line:
                current_split.append(word)
            else:
                word_splits.append(current_split)
                current_line = which
                current_split = [word]

        if current_split != []:
            word_splits.append(current_split) 

        word_splits = [' '.join(split) for split in word_splits]
        splitStr = '\n'.join(word_splits)
    else:
        splitStr = textStr

    textPt = subReg['Left'].center_point().annotate(splitStr)
    textPt = textPt.resize(size).rethicken(thickness).recolor(color).right(offset)
    return textPt

class TableSkeleton:
    def __init__(self, rows, cols, 
                 index_col_width = 1, header_row_height = 1, ### heights relative to the data columns, which are 1 and 1.
                 color = const.BGR['smokey_grey'],
                 vertical_lines = False, horizontal_lines = True,
                 render_height = 100, dimensions = (600, 400), resolution = 10):
        self.rows = rows
        self.cols = cols
        self.color = color
        self.vertical_lines = vertical_lines
        self.horizontal_lines = horizontal_lines
        self.render_height = render_height
        self.dimensions = dimensions
        self.resolution = resolution

        ### Rendering Camera
        tabCam = Camera.birds_eye(render_height, sensor_dims = dimensions)
        self.tabCam = tabCam

        ### figure out how to split space between index and columns, and header and rows
        ### this also ensures that the split arguments are integers
        row_frac = Fraction(header_row_height / (header_row_height + rows - 1)).limit_denominator(100)
        col_frac = Fraction(index_col_width / (index_col_width + cols - 1)).limit_denominator(100)
        
        ### Table Regions
        tabReg = tabCam.approximate_fov(render_height)
        headerReg, otherReg = tabReg.split(numerator = row_frac.numerator, denominator = row_frac.denominator, how = 'horizontal')
        rowRegs = [headerReg, *otherReg.subdivide(elements = rows - 1, how = 'horizontal').shapeList]
        
        tabRegs = []
        for row in rowRegs:
            indexReg, otherReg = row.split(numerator = col_frac.numerator, denominator = col_frac.denominator, how = 'vertical')
            colRegs = [indexReg, *otherReg.subdivide(elements = cols - 1, how = 'vertical').shapeList]
            tabRegs.append(colRegs)

        self.tabRegs = tabRegs ### these are lists of lists of regions! cannot be turned to Regions3D

        ### Table Lines
        self.vertLns = []
        self.horiLns = []

        if horizontal_lines:
            for row in rowRegs[:-1]: ### skip the last, cause we take the bottom points to make the lines, so we don't have line at the bottom
                skLn = Line3D(row.p3, row.p4, color = color)
                self.horiLns.append(skLn)

        if vertical_lines:
            colRegs = tabReg.subdivide(elements = self.cols, how = 'vertical').shapeList
            for col in colRegs[:-1]:
                skLn = Line3D(col.p2, col.p4, color = color)
                self.vertLns.append(skLn)

        self.vertLns = Lines3D(self.vertLns)
        self.horiLns = Lines3D(self.horiLns)

    def __getitem__(self, item):
        row, col = item
        return self.tabRegs[row][col]

    def get_lines(self):
        return self.vertLns.join(self.horiLns)

    def render(self):
        allLns = self.get_lines()
        tabScn = Scene(allLns)

        tabView = View.make(self.tabCam, tabScn, resolution = self.resolution)
        tabFrm = tabView.snap()
        return tabFrm

class Table:
    def __init__(self, table_data, header_offset = 0, index_offset = 0, data_offset = 0,
                 index_col_width = 1, header_row_height = 1,
                 index_rename_dict = None, columns_rename_dict = None,
                 number_format = '.2f', skip_after_how_many_symbols = None,
                 line_color = const.BGR['smokey_grey'], vertical_lines = False, horizontal_lines = True,
                 render_height = 100, dimensions = (600, 400), resolution = 10):
        ### table data: a row_first dictionary of dictionaries
        index = list(table_data.keys())
        all_columns = [list(table_data[i].keys()) for i in index]
        columns = all_columns[0]

        if not all([columns == col for col in all_columns]):
            raise Exception(f'Ambiguous rows in table_data: {columns}')
        
        if index_rename_dict is not None:
            index = [index_rename_dict[i] for i in index]
        if columns_rename_dict is not None:
            columns = [columns_rename_dict[c] for c in columns]

        total_rows = len(index) + 1
        total_cols = len(columns) + 1       

        self.index = index
        self.columns = columns
        self.data = [[col_val for col_val in row.values()] for row in table_data.values()]

        tabSket = TableSkeleton(total_rows, total_cols, color = line_color, vertical_lines = vertical_lines, horizontal_lines = horizontal_lines,
                                index_col_width = index_col_width, header_row_height = header_row_height,
                                render_height = render_height, dimensions = dimensions, resolution = resolution)
        
        self.tabSket = tabSket
        self.tabCam = self.tabSket.tabCam
        self.resolution = resolution
        
        self.header_offset = header_offset
        self.index_offset = index_offset
        self.data_offset = data_offset
        self.number_format = number_format
        self.skip_after_how_many_symbols = skip_after_how_many_symbols

    def get_lines(self):
        return self.tabSket.get_lines()
    
    def __getitem__(self, item):
        return self.tabSket.__getitem__(item)

    def render(self):
        allLns = self.get_lines()
        tabScn = Scene(allLns)

        annPnts = []

        ### render indices:
        for i, text in enumerate(self.index):
            myReg = self.tabSket[i + 1, 0]
            textPt = make_text_in(myReg, text, number_format = self.number_format, skip_after_how_many_symbols = self.skip_after_how_many_symbols,
                                  offset = self.index_offset)
            annPnts.append(textPt)

        ### render columns:
        for i, text in enumerate(self.columns):
            myReg = self.tabSket[0, i + 1]
            textPt = make_text_in(myReg, text, number_format = self.number_format, skip_after_how_many_symbols = self.skip_after_how_many_symbols,
                                  offset = self.header_offset)
            annPnts.append(textPt)

        ### render data:
        for i, row in enumerate(self.data):
            for j, text in enumerate(row):
                myReg = self.tabSket[i + 1, j + 1]
                textPt = make_text_in(myReg, text, number_format = self.number_format, skip_after_how_many_symbols = self.skip_after_how_many_symbols,
                                      offset = self.data_offset)
                annPnts.append(textPt)

        tabView = View.make(self.tabCam, tabScn, resolution = self.resolution, annotatedPoints = annPnts)
        tabFrm = tabView.snap()
        return tabFrm

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

    def __getitem__(self, item):
        row, col = item
        return self.raster[row][col]
    
    def __setitem__(self, item, value):
        row, col = item
        self.raster[row][col] = value

    def add(self, someFrms, row, col,
            set_size = False): ### added element overwrites the size
        
        if isinstance(someFrms, Frame):
            someFrms = BGRFrames.from_frame(someFrms, 1) 

        self[row, col] = someFrms

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


if __name__ == '__main__':
    
    chips_naive = {'total': np.random.random(), 'make': np.random.random(), 'cond': np.random.random()} 
    chips_smart = {'total': np.random.random(), 'make': np.random.random(), 'cond': np.random.random()}
    tops_smart = {'total': np.random.random(), 'make': np.random.random(), 'cond': np.random.random()}

    table_data = {'Chipped Return': chips_naive, 'Chipped Return Choice': chips_smart, 'Topspin Return': tops_smart}

    Table(table_data, number_format = '.1%', skip_after_how_many_symbols = 15, index_offset = 0.25, index_col_width = 2.5).render().show()
