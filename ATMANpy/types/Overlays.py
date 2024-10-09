
import numpy as np

from ATMANpy.types.Shapes3D import Region3D, Regions3D, AnnotatedPoint3D, Line3D, Lines3D, Point3D
from ATMANpy.types.Visual3D import DummyScene, Scene, Camera, View
from ATMANpy.types.ShapesTennis3D import Arrow

from ATMANpy.types.Frames import Frame, BGRFrame
from ATMANpy.types.MultiFrames import BGRFrames

import ATMANpy.auxil.constants as const
import ATMANpy.funcs.util_funcs as ut

from copy import deepcopy
from dataclasses import dataclass

from fractions import Fraction

def maybe_data_to_text(data_or_string, number_format, target_len = None):
    if isinstance(data_or_string, str):
        myStr = data_or_string
    else:
        myStr = f'{data_or_string:{number_format}}'

    if target_len is None:
        return myStr
    else:
        return f'{myStr:>{target_len}}'

def make_region_in(subReg, offset = 0.1, color = const.BGR['smokey_grey']):
    assert offset < 0.5
    myReg = subReg.rescale(1 - offset * 2)
    myReg = myReg.recolor(color)
    return myReg

def make_text_in(subReg, data_or_string, target_len = None,
                 size = 1, thickness = 1, color = const.BGR['leafy_green'], 
                 horizontal_offset = 0, vertical_offset = 0, skip_after_how_many_symbols = None,
                 number_format = '.2f'):
    textStr = maybe_data_to_text(data_or_string, number_format, target_len = target_len)

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

    textPt = textPt.resize(size).rethicken(thickness).recolor(color)
    textPt = textPt.right(horizontal_offset).back(vertical_offset)
    return textPt

class LineChart:
    def __init__(self, xs, ys, color = const.BGR['intense_red'], dashed = False, x_labels = None,
                 annotate_line_heights = False):
        if len(xs) != len(ys):
            raise Exception(f'x-values and y-values need to be of same length!')
        
        self.xs = np.array(xs)
        self.ys = np.array(ys)
        self.color = color
        self.dashed = dashed ### TODO: inactive for the moment!
        self.x_labels = x_labels
        self.annotate_line_heights = annotate_line_heights
        
    def __len__(self):
        return len(self.xs)
    
    def get_leftmost_value(self):
        return min(self.xs)
    
    def get_rightmost_value(self):
        return max(self.xs)
    
    def get_highest_value(self):
        return max(self.ys)
    
class ScatterChart(LineChart):
    pass ### for the moment, ScatterChart is essentially a boolean on LineChart saying: scatter, don't draw lines.

class BarChart:
    def __init__(self, xs, ys, color = const.BGR['intense_red'], 
                 x_labels = None, annotate_bar_heights = False,
                 internal_gap = 0, external_gap = 0.1):
        if len(xs) != len(ys):
            raise Exception(f'x-values and y-values need to be of same length!')
        
        self.xs = np.array(xs)
        self.ys = np.array(ys)
        self.color = color
        self.x_labels = x_labels
        self.annotate_bar_heights = annotate_bar_heights
        self.internal_gap = internal_gap
        self.external_gap = external_gap

    def __len__(self):
        return len(self.xs)
    
    def get_leftmost_value(self):
        min_x = self.xs[0]
        bar_width_unadj = (self.xs[1] - self.xs[0])
        return min_x - bar_width_unadj / 2
    
    def get_rightmost_value(self):
        max_x = self.xs[-1] ### TODO: ensure xs and ys are sorted!
        bar_width_unadj = (self.xs[-1] - self.xs[-2]) ### TODO: consider here external and internal gap?!
        return max_x + bar_width_unadj / 2
    
    def get_highest_value(self):
        return max(self.ys)

class Graph:
    def __init__(self, charts, 
                 as_density = False, ### if we plot as density, there are no y labels, and we norm y-axis accordingly.
                 color = const.BGR['smokey_grey'], offset = 0.2,
                 make_default_x_labels = True, 
                 x_label_dist = 0.5, y_label_dist = 0.5, x_label_left_shift = 0, x_tick_label_left_shift = 0,
                 x_label_format = '.0f', y_label_format = '.0f',
                 x_label = None, y_label = None, 
                 label_size = 1, tick_label_size = 1,
                 header_text = 'Chart', header_horizontal_offset = 8, header_vertical_offset = 1,
                 header_height = 1, graph_height = 9,
                 render_height = 100, dimensions = (600, 400), resolution = 10):
        ### TODO: add axis labels (y-axis vertical! how?)
        self.charts = charts
        self.color = color

        self.as_density = as_density
        self.make_default_x_labels = make_default_x_labels
        self.x_label_format = x_label_format
        self.y_label_format = y_label_format

        self.x_label = x_label
        self.y_label = y_label

        self.header_text = header_text
        self.header_horizontal_offset = header_horizontal_offset
        self.header_vertical_offset = header_vertical_offset

        self.x_label_dist = x_label_dist
        self.y_label_dist = y_label_dist

        self.x_label_left_shift = x_label_left_shift
        self.x_tick_label_left_shift = x_tick_label_left_shift

        self.label_size = label_size
        self.tick_label_size = tick_label_size

        self.render_height = render_height
        self.dimensions = dimensions
        self.resolution = resolution

        ### Rendering Camera
        cam = Camera.birds_eye(render_height, sensor_dims = dimensions)
        self.cam = cam

        ### Graph Regions
        camReg = cam.approximate_fov(render_height)
        headerReg, otherReg = camReg.split(numerator = header_height, denominator = graph_height, how = 'horizontal')
        self.headerReg = headerReg
        self.otherReg = otherReg

        ### Graph Axes
        graphReg = otherReg.rescale(1 - offset)
        x_axis, y_axis = graphReg['Bottom'], graphReg['Left']
        self.x_axis = x_axis.recolor(color)
        self.y_axis = y_axis.recolor(color)

        self.x_orig, self.y_orig, _ = x_axis.leftPt.coords
        self.x_len = x_axis.length()
        self.y_len = y_axis.length()

        self.renorm_axes()

    def renorm_axes(self):
        if len(self.charts) == 0:
            min_x = 0
            min_y = 0
            max_x = 1
            max_y = 1
        else:
            lfmosts = [chart.get_leftmost_value() for chart in self.charts]
            rtmosts = [chart.get_rightmost_value() for chart in self.charts]
            himosts = [chart.get_highest_value() for chart in self.charts]

            min_x = 0 #min(0, min(lfmosts))    ### never start an x after 0
            max_x = max(rtmosts)
            min_y = 0                       ### always start y on 0
            max_y = max(himosts)

            ### rounding to nearest integer TODO: make more dynamic?
            min_x = int(np.floor(min_x))
            max_x = int(np.ceil(max_x))
            min_y = int(np.floor(min_y))
            if self.as_density:
                max_y = 1.05 * max_y ### scale up a bit
            else:
                max_y = int(np.ceil(max_y))

        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def add(self, chart):
        self.charts = [*self.charts, chart]
        self.renorm_axes()

    def preview(self):
        ### Make Axes
        myScn = Scene([self.x_axis, self.y_axis])

        min_x, max_x, min_y, max_y = self.min_x, self.max_x, self.min_y, self.max_y
        
        ### Make Axis Tick Labels
        if self.make_default_x_labels:
            min_x_str = maybe_data_to_text(min_x, self.x_label_format)
            max_x_str = maybe_data_to_text(max_x, self.x_label_format)

            min_x_pt = self.x_axis.leftPt.back(self.x_label_dist).annotate(min_x_str).resize(self.tick_label_size)
            max_x_pt = self.x_axis.rightPt.back(self.x_label_dist).annotate(max_x_str).resize(self.tick_label_size)
            x_labels = [min_x_pt, max_x_pt]
        else:
            x_labels = []

        if not self.as_density: 
            min_y_str = maybe_data_to_text(min_y, self.y_label_format)
            max_y_str = maybe_data_to_text(max_y, self.y_label_format)

            min_y_pt = self.y_axis.rightPt.left(self.y_label_dist).annotate(min_y_str).resize(self.tick_label_size)
            max_y_pt = self.y_axis.leftPt.left(self.y_label_dist).annotate(max_y_str).resize(self.tick_label_size)
            y_labels = [min_y_pt, max_y_pt]
        else:
            y_labels = []

        ### Make Axis Labels
        axis_labels = []
        if self.x_label is not None:
            x_label_pt = self.x_axis.center_point().back(self.x_label_dist * 2).left(self.x_label_left_shift).annotate(self.x_label).resize(self.label_size)
            axis_labels.append(x_label_pt)
        if self.y_label is not None:
            y_label_pt = self.y_axis.center_point().left(self.y_label_dist * 2).annotate(self.y_label).resize(self.label_size)
            axis_labels.append(y_label_pt)

        ### Make Header Point
        if self.header_text != '':
            headerPt = make_text_in(self.headerReg, self.header_text, color = self.color, 
                                    horizontal_offset = self.header_horizontal_offset,
                                    vertical_offset = self.header_vertical_offset)
            header_pnts = [headerPt]
        else:
            header_pnts = []

        x_label_pnts = []
        val_label_pnts = []
        for chart in self.charts:
            ### re-norm the xs and ys (between 0 and 1)
            xs_norm = chart.xs / (self.max_x - self.min_x)
            ys_norm = chart.ys / (self.max_y - self.min_y)

            ### bring normalized values to coordinates
            xs_rect = self.x_len * xs_norm + self.x_orig
            ys_rect = -(self.y_len * ys_norm) + self.y_orig ### need minus since y coordinates go up to down
            zs_zero = np.zeros(len(chart))
            
            coordArr = np.array([xs_rect, ys_rect, zs_zero])

            if isinstance(chart, ScatterChart):
                coordPt_vectorized = Point3D(coordArr, color = chart.color)
                myScn = myScn.put(coordPt_vectorized)
            elif isinstance(chart, LineChart):
                coordArr_prev = coordArr[:, :-1]
                coordArr_next = ut.shift_array(coordArr, -1, axis = 1)[:, :-1]

                coordPt_vectorized = Point3D(coordArr_prev)
                coordPt_vectorized_next = Point3D(coordArr_next)
                coordLn_vectorized = Line3D(coordPt_vectorized, coordPt_vectorized_next, color = chart.color)
                myScn = myScn.put(coordLn_vectorized)
                
                ### Make x-labels
                if chart.x_labels is not None:
                    for coords, label in zip(coordArr.T, chart.x_labels):
                        x, y, z = coords
                        coords_base = (x, self.y_orig, z)

                        labelStr = maybe_data_to_text(label, '.0f')
                        anchorPnt = Point3D(coords_base).back(self.x_label_dist).left(self.x_tick_label_left_shift)
                        annPnt = anchorPnt.annotate(labelStr).resize(self.tick_label_size)
                        x_label_pnts.append(annPnt)

                ### Make annotations for values
                if chart.annotate_line_heights:
                    for coords, y_val in zip(coordArr.T, chart.ys):
                        x, y, z = coords

                        labelStr = maybe_data_to_text(y_val, number_format = self.y_label_format)
                        anchorPnt = Point3D(coords).front(self.x_label_dist).left(self.x_tick_label_left_shift)
                        annPnt = anchorPnt.recolor(chart.color).annotate(labelStr)
                        val_label_pnts.append(annPnt)

            elif isinstance(chart, BarChart):
                xs = coordArr[0, :-1]
                xs_next = ut.shift_array(coordArr, -1, axis = 1)[0, :-1]

                bar_half_width = (xs_next - xs) / 2
                bar_half_width_adj = bar_half_width * (1 - chart.external_gap)
                bar_half_width_adj = np.append(bar_half_width_adj, bar_half_width_adj[-1])

                ### Make Regions
                regions = []
                for coords, hw in zip(coordArr.T, bar_half_width_adj):
                    x, y, z = coords
                    x1, x2 = x - hw, x + hw
                    y1, y2 = self.y_orig, y

                    p1, p2, p3, p4 = Point3D((x1, y1, z)), Point3D((x1, y2, z)), Point3D((x2, y2, z)), Point3D((x2, y1, z))
                    barReg = Region3D.from_points3D(p1, p2, p3, p4, color = chart.color)

                    regions.append(barReg)
                
                myScn = myScn.put(regions)

                ### Make x-labels for Bar Regions
                if chart.x_labels is not None:
                    for coords, label in zip(coordArr.T, chart.x_labels):
                        x, y, z = coords
                        coords_base = (x, self.y_orig, z)

                        labelStr = maybe_data_to_text(label, '.0f')
                        anchorPnt = Point3D(coords_base).back(self.x_label_dist).left(self.x_tick_label_left_shift)
                        annPnt = anchorPnt.annotate(labelStr)
                        x_label_pnts.append(annPnt)
                    
                ### Make annotations for Bar Regions (if necessary)
                if chart.annotate_bar_heights:
                    for coords, y_val in zip(coordArr.T, chart.ys):
                        x, y, z = coords

                        labelStr = maybe_data_to_text(y_val, number_format = self.y_label_format)
                        anchorPnt = Point3D(coords).front(self.x_label_dist).left(self.x_tick_label_left_shift)
                        annPnt = anchorPnt.recolor(chart.color).annotate(labelStr)
                        val_label_pnts.append(annPnt)

        annotatedPoints = [*x_labels, *y_labels, *header_pnts, *x_label_pnts, *val_label_pnts, *axis_labels]
        return myScn, annotatedPoints ### by exposing these we can morph and move chart elements.
    
    def render(self):
        myScn, annotatedPoints = self.preview()
        axView = View.make(self.cam, myScn, resolution = self.resolution, annotatedPoints = annotatedPoints)
        axFrm = axView.snap()
        return axFrm

class TableSkeleton:
    def __init__(self, rows, cols, 
                 shrinkage = 0, horizontal_offset = 0, vertical_offset = 0, 
                 index_col_width = 1, header_row_height = 1, ### heights relative to the data columns, which are 1 and 1.
                 color = const.BGR['smokey_grey'],
                 vertical_lines = False, horizontal_lines = True,
                 truncate_vertical_lines_by = 0, truncate_horizontal_lines_by = 0,
                 index_line_thickness = 0, header_line_thickness = 0, data_line_thickness = 0,
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
        tabReg = tabReg.rescale(1 - shrinkage).back(vertical_offset).right(horizontal_offset)
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

        for row_idx, rowRegs in enumerate(tabRegs):
            for col_idx, tabReg in enumerate(rowRegs):
                if horizontal_lines:
                    if row_idx < self.rows - 1 and col_idx < self.cols - truncate_horizontal_lines_by:           
                        thickness = header_line_thickness if row_idx == 0 else data_line_thickness             
                        skLn = Line3D(tabReg.p3, tabReg.p4, color = color, thickness = thickness)
                        self.horiLns.append(skLn)
                
                if vertical_lines:
                    if col_idx < self.cols - 1 and row_idx < self.rows - truncate_vertical_lines_by:        
                        thickness = index_line_thickness if col_idx == 0 else data_line_thickness
                        skLn = Line3D(tabReg.p2, tabReg.p4, color = color, thickness = thickness)
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
    def __init__(self, table_data, 
                 title = None, title_size = 2, title_thickness = 1, title_color = const.BGR['leafy_green'],
                 title_horizontal_offset = 0, title_vertical_offset = 0,
                 skeleton_shrinkage = 0, skeleton_horizontal_offset = 0, skeleton_vertical_offset = 0,
                 header_offset = 0, index_offset = 0, data_offset = 0,
                 index_col_width = 1, header_row_height = 1,
                 index_line_thickness = 0, header_line_thickness = 0, data_line_thickness = 0,
                 index_rename_dict = None, columns_rename_dict = None,
                 null_rows = 0, null_columns = 0,
                 data_right_adjust = False, ### only works if there are no lineskips in columns...
                 header_text_size = 1, index_text_size = 1, data_text_size = 1,
                 header_text_thickness = 1, index_text_thickness = 1, data_text_thickness = 1, 
                 header_text_color = const.BGR['leafy_green'], index_text_color = const.BGR['leafy_green'], data_text_color = const.BGR['leafy_green'],
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

        total_rows = len(index) + null_rows + 1
        total_cols = len(columns) + null_columns + 1       

        self.index = index
        self.columns = columns
        self.data = [[col_val for col_val in row.values()] for row in table_data.values()]

        tabSket = TableSkeleton(total_rows, total_cols, color = line_color, 
                                shrinkage = skeleton_shrinkage, horizontal_offset = skeleton_horizontal_offset, vertical_offset = skeleton_vertical_offset,
                                vertical_lines = vertical_lines, horizontal_lines = horizontal_lines,
                                index_col_width = index_col_width, header_row_height = header_row_height,
                                index_line_thickness = index_line_thickness, header_line_thickness = header_line_thickness, data_line_thickness = data_line_thickness,
                                truncate_horizontal_lines_by = null_columns, truncate_vertical_lines_by = null_rows,
                                render_height = render_height, dimensions = dimensions, resolution = resolution)
        
        self.tabSket = tabSket
        self.tabCam = self.tabSket.tabCam
        self.resolution = resolution
        
        self.header_offset = header_offset
        self.index_offset = index_offset
        self.data_offset = data_offset
        self.data_right_adjust = data_right_adjust
        self.number_format = number_format
        self.skip_after_how_many_symbols = skip_after_how_many_symbols

        self.header_text_size = header_text_size
        self.index_text_size = index_text_size
        self.data_text_size = data_text_size

        self.header_text_thickness = header_text_thickness
        self.index_text_thickness = index_text_thickness
        self.data_text_thickness = data_text_thickness

        self.header_text_color = header_text_color
        self.index_text_color = index_text_color
        self.data_text_color = data_text_color
        
        self.title = title
        self.title_size = title_size
        self.title_color = title_color
        self.title_thickness = title_thickness
        self.title_horizontal_offset = title_horizontal_offset
        self.title_vertical_offset = title_vertical_offset

        if title is None:
            self.headerReg = None
        else:
            tabReg = self.tabCam.approximate_fov(render_height)
            self.headerReg, _ = tabReg.split(numerator = 1, denominator = total_rows, how = 'horizontal')

    def get_lines(self):
        return self.tabSket.get_lines()
    
    def __getitem__(self, item):
        return self.tabSket.__getitem__(item)

    def render(self):
        allLns = self.get_lines()
        tabScn = Scene(allLns)

        annPnts = []

        ### render title:
        if self.title is not None:
            textPt = make_text_in(self.headerReg, self.title, 
                                  size = self.title_size, color = self.title_color, thickness = self.title_thickness, 
                                  horizontal_offset = self.title_horizontal_offset, vertical_offset = self.title_vertical_offset)
            annPnts.append(textPt)

        ### render indices:
        for i, text in enumerate(self.index):
            myReg = self.tabSket[i + 1, 0]
            textPt = make_text_in(myReg, text, number_format = self.number_format, 
                                  size = self.index_text_size, thickness = self.index_text_thickness, color = self.index_text_color, 
                                  skip_after_how_many_symbols = self.skip_after_how_many_symbols,
                                  horizontal_offset = self.index_offset)
            annPnts.append(textPt)

        ### render columns:
        for i, text in enumerate(self.columns):
            myReg = self.tabSket[0, i + 1]
            textPt = make_text_in(myReg, text, number_format = self.number_format, 
                                  size = self.header_text_size, thickness = self.header_text_thickness, color = self.header_text_color,
                                  skip_after_how_many_symbols = self.skip_after_how_many_symbols,
                                  horizontal_offset = self.header_offset)
            annPnts.append(textPt)

        ### render data:
        for i, row in enumerate(self.data):
            for j, text in enumerate(row):
                if self.data_right_adjust:
                    target_len = len(self.columns[j])
                else:
                    target_len = None

                myReg = self.tabSket[i + 1, j + 1]
                textPt = make_text_in(myReg, text, number_format = self.number_format, target_len = target_len,
                                      size = self.data_text_size, thickness = self.data_text_thickness, color = self.data_text_color,
                                      skip_after_how_many_symbols = self.skip_after_how_many_symbols,
                                      horizontal_offset = self.data_offset)
                annPnts.append(textPt)

        tabView = View.make(self.tabCam, tabScn, resolution = self.resolution, annotatedPoints = annPnts)
        tabFrm = tabView.snap()
        return tabFrm

class DiagramShape:
    def __init__(self, text, contReg, shape_offset = 0.1, shape_color = const.BGR['smokey_grey'],
                 text_size = 1, text_thickness = 1, text_color = const.BGR['leafy_green'], 
                 text_horizontal_offset = 0, text_vertical_offset = 0, 
                 text_skip_after_how_many_symbols = None):
        self.text = text
        self.contReg = contReg

        self.shape_offset = shape_offset
        self.shape_color = shape_color

        self.text_size = text_size
        self.text_thickness = text_thickness
        self.text_color = text_color
        self.text_horizontal_offset = text_horizontal_offset
        self.text_vertical_offset = text_vertical_offset
        self.text_skip_after_how_many_symbols = text_skip_after_how_many_symbols

    def render(self):
        ### returns a region and an annotated point
        outReg = make_region_in(self.contReg, offset = self.shape_offset, color = self.shape_color)
        outPnt = make_text_in(outReg, self.text, size = self.text_size, color = self.text_color, thickness = self.text_thickness,
                              horizontal_offset = self.text_horizontal_offset, vertical_offset = self.text_vertical_offset,
                              skip_after_how_many_symbols = self.text_skip_after_how_many_symbols)
        return outReg, outPnt

class Diagram:
    def __init__(self, diagram_text_array, shape_offset = 0.1, shape_color = const.BGR['smokey_grey'],
                 text_size = 1, text_thickness = 1, text_color = const.BGR['leafy_green'], 
                 text_horizontal_offset = 0, text_vertical_offset = 0,
                 text_skip_after_how_many_symbols = None,
                 arrow_gamma = np.pi / 4, arrow_theta = 0, arrow_hat_length = 1, 
                 arrow_color = const.BGR['strong_blue'], arrow_alpha = 1, arrow_thickness = 0,
                 render_height = 100, dimensions = (600, 400), resolution = 10):
        ### diagram text array is an irregular array (an arbitrary list of lists) 
        # that contains the text that is to be printed in the diagram shapes.

        # determine size of the Table Skeleton needed for the array.
        self.rows = len(diagram_text_array)
        self.cols = max([len(row) for row in diagram_text_array])

        self.tabSket = TableSkeleton(self.rows, self.cols, horizontal_lines = False, vertical_lines = False,
                                     render_height = render_height, dimensions = dimensions, resolution = resolution)
        self.tabCam = self.tabSket.tabCam
        self.resolution = resolution

        ### make the shapes
        self.shapes = []
        for row_idx in range(self.rows):
            row_elements = len(diagram_text_array[row_idx])
            if row_elements == 0:
                continue

            row_shapes = []
            for col_idx in range(self.cols):
                ### for irregular diagram_text_arrays, we need to distinguish
                if col_idx < row_elements: ### everything to the left is a valid shape ...
                    text = diagram_text_array[row_idx][col_idx]
                    contReg = self.tabSket[row_idx, col_idx]
                    prev_p1, prev_p3 = contReg.p1, contReg.p3

                    myShape = DiagramShape(text, contReg, shape_offset = shape_offset, shape_color = shape_color,
                                           text_size = text_size, text_thickness = text_thickness, text_color = text_color,
                                           text_horizontal_offset = text_horizontal_offset, text_vertical_offset = text_vertical_offset, 
                                           text_skip_after_how_many_symbols = text_skip_after_how_many_symbols)
                    row_shapes.append(myShape)
                    
                else: ### ... whereas for everything to the right, we enlarge the last shape
                    contReg = self.tabSket[row_idx, col_idx]
                    next_p2, next_p4 = contReg.p2, contReg.p4
                    myShape.contReg = Region3D.from_points3D(prev_p1, next_p2, prev_p3, next_p4) ### modify the previous shape.

                    row_shapes.append(myShape)

            self.shapes.append(row_shapes)
            ### -> we can now assume that the shape array is semi-regular: it contains full size rows or empty rows.

        non_empty_row_idxs = [idx for idx, row in enumerate(self.shapes) if row != []]

        ### make the arrows
        # arrows go up to down, so we make them column wise
        self.arrows = []
        for row_idx in range(self.rows):
            non_empty_rows_below_idxs = [i for i in non_empty_row_idxs if i > row_idx]
            row_is_empty = row_idx not in non_empty_row_idxs
            row_is_last = len(non_empty_rows_below_idxs) == 0

            row_arrows = []
            for col_idx in range(self.cols):
                if not row_is_empty and not row_is_last:
                    origin_region = self.shapes[row_idx][col_idx].contReg
                    origin_point = origin_region['Bottom'].center_point()
                    o_x, o_y, o_z = origin_point.coords

                    # find target shape
                    target_row_idx = non_empty_rows_below_idxs[0]
                    target_region = self.shapes[target_row_idx][col_idx].contReg
                    t_y = target_region['Top'].center_point().y

                    target_point = Point3D((o_x, t_y, o_z))

                    if origin_point == target_point:
                        row_arrows.append(None)
                    else:
                        myArrow = Arrow.make(origin_point, target_point, 
                                             gamma = arrow_gamma, theta = arrow_theta, hat_length = arrow_hat_length,
                                             color = arrow_color, alpha = arrow_alpha, thickness = arrow_thickness)
                        row_arrows.append(myArrow)
                else:
                    row_arrows.append(None)
            
            self.arrows.append(row_arrows)

    def __getitem__(self, item):
        row, col = item
        return self.shapes[row][col], self.arrows[row][col]

    def __setitem__(self, item, shape_arrow_pair):
        shape, arrow = shape_arrow_pair
        row, col = item
        self.shapes[row][col] = shape
        self.arrows[row][col] = arrow

    def deepcopy(self):
        return deepcopy(self)

    def render(self):
        diaRegs = []
        annPnts = []
        diagArrows = []

        for row in self.shapes:
            for diagShape in row:
                if diagShape is not None:
                    outReg, outPnt = diagShape.render()
                    diaRegs.append(outReg)
                    annPnts.append(outPnt)

        ### technically, we can have the same shape multiple times and the same annPnt multiple times 
        # because the row_shapes may contain duplicate references to the same shape!
        # alleviate by filtering duplicate annotated points and shapes

        for row in self.arrows:
            for arrow in row:
                if arrow is not None:
                    diagArrows.append(arrow)

        myScn = Scene(diaRegs).put(diagArrows)
        myView = View.make(self.tabCam, myScn, annotatedPoints = annPnts, resolution = self.resolution)
        myFrm = myView.snap()

        return myFrm

class LegendRow:
    def __init__(self, text = None, shape_color = None,
                 text_color = const.BGR['bright_grey'], text_size = 1, text_thickness = 1, 
                 text_horizontal_offset = 0, text_vertical_offset = 0,
                 shape_offset = 0.1):
        self.text = text
        self.text_color = text_color
        self.text_size = text_size
        self.text_thickness = text_thickness
        self.text_horizontal_offset = text_horizontal_offset
        self.text_vertical_offset = text_vertical_offset

        self.shape_color = shape_color
        self.shape_offset = shape_offset

        self.is_null = text is None

class Legend:
    def __init__(self, legend_rows, legend_dimensions = (600, 400), 
                 render_height = 100, resolution = 10, split_denominator = 4, borders = True,
                 header_text = 'Legend', header_size = 2, header_thickness = 2, header_offset = 10, header_color = const.BGR['bright_grey']):
        self.legend_rows = legend_rows

        self.legend_dimensions = legend_dimensions
        self.render_height = render_height
        self.resolution = resolution

        self.split_denominator = split_denominator
        self.borders = borders

        self.header_text = header_text
        self.header_size = header_size
        self.header_thickness = header_thickness
        self.header_offset = header_offset
        self.header_color = header_color

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
    
    def add(self, legend_row):
        return Legend([*self.legend_rows, legend_row], 
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

        num_els = len(self.legend_rows)

        if self.header_text != '':
            subRegs = legReg.subdivide(elements = num_els + 1, how = 'horizontal')
            headerReg, otherRegs = subRegs[0], subRegs[1:]

            headerPt = make_text_in(headerReg, self.header_text, 
                                    size = self.header_size, 
                                    thickness = self.header_thickness,
                                    horizontal_offset = self.header_offset,
                                    color = self.header_color)

            legend_shapes, annotated_points = [legLines], [headerPt]
        else:
            otherRegs = legReg.subdivide(elements = num_els + 1, how = 'horizontal')
            legend_shapes, annotated_points = [], []
        
        for row, subReg in zip(self.legend_rows, otherRegs):
            
            if row.is_null: ### this is so we can have a legend of fixed subdivisions with empty elements -> None as a placeholder
                continue

            subReg_left, subReg_right = subReg.split(numerator = 1, denominator = self.split_denominator, how = 'vertical')

            ### shape on the left side of the legend element
            try:
                shape_color = row.shape_color
                shape_offset = row.shape_offset
            except KeyError:
                pass
            else:
                shapeReg = make_region_in(subReg_left, offset = shape_offset, color = shape_color)
                legend_shapes.append(shapeReg)

            # ### text to add on the left side of the legend element, on top of shape potentially
            # try:
            #     text_string_shape = params['text_string_shape']
            #     text_color_shape = params['text_color_shape']
            #     text_size_shape = params['text_size_shape']
            #     text_thickness_shape = params['text_thickness_shape']
            #     text_offset_shape = params['text_offset_shape']
            # except KeyError:
            #     pass
            # else:
            #     textPt_left = make_text_in(subReg_left, text_string_shape, 
            #                                size = text_size_shape, thickness = text_thickness_shape, 
            #                                color = text_color_shape, horizontal_offset = text_offset_shape)
            #     annotated_points.append(textPt_left)

            ### text on the right side of the legend element
            try:
                text_string = row.text
                text_color = row.text_color
                text_size = row.text_size
                text_thickness = row.text_thickness
                text_horizontal_offset = row.text_horizontal_offset
                text_vertical_offset = row.text_vertical_offset
            except KeyError:
                pass
            else:
                textPt_right = make_text_in(subReg_right, text_string, 
                                            size = text_size, thickness = text_thickness, color = text_color, 
                                            horizontal_offset = text_horizontal_offset, vertical_offset = text_vertical_offset)
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

class FrameRaster: ### a tiling of a Frame with smaller Frames, in rows and columns
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        self.raster = np.full((rows, cols), fill_value = None).tolist()
        self.element_dims = None
        self.output_dims = None

    def __getitem__(self, item):
        row, col = item
        return self.raster[row][col]

    def __setitem__(self, item, value):
        row, col = item
        self.raster[row][col] = value    
    
    def deepcopy(self):
        return deepcopy(self)

    def add(self, someFrm, row, col): ### added element overwrites the size
        dims = someFrm.cols, someFrm.rows ### note the switch here.
        if self.element_dims is not None:
            if dims != self.element_dims:
                raise Exception(f'Raster element dimensions ({self.element_dims}) have to match input frame dimensions ({dims})')
        else:
            self.element_dims = dims

        if self.output_dims is None:
            el_cols, el_rows = self.element_dims
            self.output_dims = el_cols * self.cols, el_rows * self.rows
        
        self[row, col] = someFrm
        
    def render(self):
        ### idea: get all the element MultiFrames to have the target dimensions and the target length
        ### for None elements, create black 
        ### then, enlarge them, with padding corresponding to where they are placed in the raster
        ### then, simply add these new MultiFrames together.
        el_cols, el_rows = self.element_dims
        al_cols, al_rows = self.output_dims

        jointArr = np.zeros((al_rows, al_cols, 3), dtype = 'uint8')

        for row_idx, rows in enumerate(self.raster):
            for col_idx, someFrm in enumerate(rows):
                if someFrm is None:
                    continue
                
                addArr = someFrm.array.copy()
                
                ### find right spot in joint array
                row_left, row_right = row_idx * el_rows, (row_idx + 1) * el_rows
                col_left, col_right = col_idx * el_cols, (col_idx + 1) * el_cols
                jointArr[row_left:row_right, col_left:col_right] = addArr

        return BGRFrame(jointArr)
    
class FramesRaster: ### a tiling of MultiFrames with smaller MultiFrames, in rows and columns
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

    def deepcopy(self):
        return deepcopy(self)

    def to_frame_raster(self, t = -1):
        ### get a specific frame t and return a corresponding FrameRaster
        frmRst = FrameRaster(self.rows, self.cols)

        if self.length == 0:
            return frmRst

        for row_idx in range(self.rows):
            for col_idx in range(self.cols):
                myFrms = self[row_idx, col_idx]

                if myFrms is not None:
                    if t < -myFrms.length():
                        t_adj = 0
                    elif -myFrms.length() <= t < 0:
                        t_adj = t
                    elif 0 <= t < myFrms.length():
                        t_adj = t
                    elif myFrms.length() <= t:
                        t_adj = -1
                    else:
                        raise Exception(f'Unhandled case: t = {t}, length of Frames = {myFrms.length()}')

                    myFrm = myFrms[t_adj]

                    frmRst.add(myFrm, row_idx, col_idx)

        return frmRst

    def add(self, someFrms, row, col,
            set_size = False): ### added element overwrites the size
        
        if isinstance(someFrms, Frame):
            someFrms = BGRFrames.from_frame(someFrms, 1) 

        dims = someFrms.cols(), someFrms.rows() ### note the switch here.
        if self.element_dims is not None:
            if dims != self.element_dims:
                raise Exception(f'Raster element dimensions ({self.element_dims}) have to match input frame dimensions ({dims})')
        else:
            self.element_dims = dims

        if someFrms.length() > self.length or set_size:
            self.length = someFrms.length()
        
        if self.output_dims is None:
            el_cols, el_rows = self.element_dims
            self.output_dims = el_cols * self.cols, el_rows * self.rows

        self[row, col] = someFrms
        
    def render(self):
        ### idea: get all the element MultiFrames to have the target dimensions and the target length
        ### for None elements, create black 
        ### then, enlarge them, with padding corresponding to where they are placed in the raster
        ### then, simply add these new MultiFrames together.
        if self.length == 0:
            print('Empty Raster, rendering None')
            return None

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

def frame_cascade(list_of_cascFrms, baseRst):
    num_rows, num_cols = baseRst.rows, baseRst.cols

    occupied_positions = []
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            if baseRst[row_idx, col_idx] is not None:
                occupied_positions.append((row_idx, col_idx))

    occupied_indices = [row * num_cols + col for row, col in occupied_positions]

    num_frms, num_occ = len(list_of_cascFrms), len(occupied_indices)

    if num_frms + num_occ != num_rows * num_cols:
        raise Exception(f'Cascade attempted with {num_frms} frames and raster of size {num_rows}x{num_cols} with {num_occ} occupied positions')

    stillFrm_row_col_truples = []
    list_of_renderedFrms = []
    for idx in range(num_frms):
        baseRst_copy = baseRst.deepcopy() ### this is the raster that will be mutated throughout this loop iteration.
        cascFrms = list_of_cascFrms[idx]

        offset = sum([idx >= occ_idx for occ_idx in occupied_indices])
        adj_idx = idx + offset
        row, col = adj_idx // num_cols, adj_idx % num_rows
        baseRst_copy.add(cascFrms, row, col)    

        for stillFrm, still_row, still_col in stillFrm_row_col_truples:
            baseRst_copy.add(stillFrm, still_row, still_col)

        renderedFrms = baseRst_copy.render()
        list_of_renderedFrms.append(renderedFrms)

        stillFrm_row_col_truples.append((cascFrms[-1], row, col))

    ### make the final raster, for returning.
    outRst = baseRst.deepcopy()
    for stillFrm, still_row, still_col in stillFrm_row_col_truples:
        outRst.add(stillFrm, still_row, still_col)

    outRst = outRst.to_frame_raster()

    ### make the rendered frames, for returning
    jntFrms = BGRFrames.from_frames(list_of_renderedFrms)
    return outRst, jntFrms

def reverse_cascade(finalRst, keep_which = None, 
                    frames_per_element = 25, acceleration = 1):
    ### TODO: finalRst has to be a FrameRaster for the moment. Generalize if needed.

    if keep_which is None:
        keep_which = []
    
    myRst = finalRst.deepcopy()
    num_rows, num_cols = finalRst.rows, finalRst.cols

    accel_index = 0
    list_of_renderedFrms = []
    for row in reversed(range(num_rows)):
        for col in reversed(range(num_cols)):
            if not (row, col) in keep_which:
                accelerated_frames_per_element = round((1 / acceleration) ** accel_index * frames_per_element)

                myRst[row, col] = None
                myFrm = myRst.render()
                myFrms = BGRFrames.from_frame(myFrm, repetitions = accelerated_frames_per_element)
                list_of_renderedFrms.append(myFrms)

            accel_index += 1

    jntFrms = BGRFrames.from_frames(list_of_renderedFrms)
    return myRst, jntFrms


if __name__ == '__main__':
    TEST_CHART = False
    TEST_TABLES = True
    TEST_RASTERS = False
    TEST_DIAGRAMS = False
    TEST_LEGEND = False

    #####################
    ### CHART TESTING ###
    #####################
    if TEST_CHART:  
        myGraph = Graph([])
        myGraph.render().show()

        aChart = LineChart([0, 1, 2, 3], [2, 3, 4, 2])
        myGraph.add(aChart)
        myGraph.render().show()

        bChart = LineChart([0, 1, 2, 3], [3, 0, 1, 1], color = const.BGR['leafy_green'])
        myGraph.add(bChart)
        myGraph.render().show()
        
        cChart = ScatterChart([0.1, 0.5, 0.9, 1.2, 2.5], [3.5, 1.5, 2.5, 0.5, 2.5], color = const.BGR['ball_yellow'])
        myGraph.add(cChart)
        myGraph.render().show()

        ### TEST BAR CHARTS
        barChart = BarChart([1, 2, 3], [2, 5, 6], external_gap = 0.5, color = const.BGR['minty_green'])
        myGraph = Graph([barChart])
        myGraph.render().show()
    
        barChart = BarChart([1, 2, 3], [2, 5, 6], external_gap = 0.5, color = const.BGR['minty_green'], 
                            x_labels = ['a', 'b', 'c'])
        myGraph = Graph([barChart], make_default_x_labels = False)
        myGraph.render().show()

        barChart = BarChart([1, 2, 3], [2, 5, 6], external_gap = 0.5, color = const.BGR['minty_green'], 
                            x_labels = ['a', 'b', 'c'], annotate_bar_heights = True)
        myGraph = Graph([barChart], make_default_x_labels = False)
        myGraph.render().show()

        barChart = BarChart([1, 2, 3, 4], [0.1, 0.6, 0.2, 0.3], external_gap = 0.5, color = const.BGR['minty_green'], 
                            x_labels = ['a', 'b', 'c', 'd'], annotate_bar_heights = True)
        myGraph = Graph([barChart], make_default_x_labels = False, y_label_format = '.0%')
        myGraph.render().show()

        ### TEST NUMBER FORMATS
        aChart = LineChart([0, 1, 2, 3], [0.1, 0.6, 0.2, 0.3])
        myGraph = Graph([aChart], y_label_format = '.0%', y_label = 'Ret. Win%', x_label = 'Rally Length', 
                            y_label_dist = 1.25, offset = 0.35, x_label_left_shift = 1.5)
        myGraph.render().show()

    #####################
    ### TABLE TESTING ###
    #####################
    if TEST_TABLES:
        chips_naive = {'total': np.random.random(), 'make': np.random.random(), 'cond': np.random.random()} 
        chips_smart = {'total': np.random.random(), 'make': np.random.random(), 'cond': np.random.random()}
        tops = {'total': np.random.random(), 'make': np.random.random(), 'cond': np.random.random()}

        table_data = {'Chipped Return': chips_naive, 'Chipped Return Choice': chips_smart, 'Topspin Return': tops}

        myTab = Table(table_data, number_format = '.1%', skip_after_how_many_symbols = 15, index_offset = 0.25, index_col_width = 2.5,
                    header_line_thickness = 0)
        myTab.render().show()
        
        chips_naive_tot = {k: v for k, v in chips_naive.items() if k == 'total'}
        chips_smart_tot = {k: v for k, v in chips_smart.items() if k == 'total'}
        tops_tot = {k: v for k, v in tops.items() if k == 'total'}
        table_data = {'Topspin Return': tops_tot, 'Chipped Return': chips_naive_tot}

        myTab = Table(table_data, 
                      number_format = '.1%', skip_after_how_many_symbols = 15, index_offset = 0.25, index_col_width = 2.5,
                    null_columns = 2, header_line_thickness = 0.2)
        myTab.render().show()
        
        myTab = Table(table_data, 
                      skeleton_shrinkage = 0.2, title = 'Some Title', title_horizontal_offset = 5,
                      number_format = '.1%', skip_after_how_many_symbols = 15, index_offset = 0.25, index_col_width = 2.5,
                    null_columns = 2, header_line_thickness = 0.2)
        myTab.render().show()

        chips_naive = {'Total Walala': np.random.random(), 'Make Waskdjh': np.random.random(), 'asd': np.random.random()} 
        chips_smart = {'Total Walala': np.random.random(), 'Make Waskdjh': np.random.random(), 'asd': np.random.random()}
        tops = {'Total Walala': np.random.random(), 'Make Waskdjh': np.random.random(), 'asd': np.random.random()}

        table_data = {'Chipped Return': chips_naive, 'Chipped Return Choice': chips_smart, 'Topspin Return': tops}
        myTab = Table(table_data, data_right_adjust = False,
                      skeleton_shrinkage = 0.1, title = 'Some Title', title_horizontal_offset = 5,
                      number_format = '.1%', skip_after_how_many_symbols = 15, index_offset = 0.25, index_col_width = 1.5,
                    null_columns = 0, header_line_thickness = 0.2)
        myTab.render().show()
        
        table_data = {'Chipped Return': chips_naive, 'Chipped Return Choice': chips_smart, 'Topspin Return': tops}
        myTab = Table(table_data, data_right_adjust = True,
                      skeleton_shrinkage = 0.1, title = 'Some Title', title_horizontal_offset = 5,
                      number_format = '.1%', skip_after_how_many_symbols = 15, index_offset = 0.25, index_col_width = 1.5,
                    null_columns = 0, header_line_thickness = 0.2)
        myTab.render().show()


    ########################
    #### RASTER TESTING ####
    ########################
    if TEST_RASTERS:
        from ATMANpy.types.Frames import BGRFrame
        from ATMANpy.types.MultiFrames import BGRFrames
        from ATMANpy.types.Streams import DecapStream

        random_arrays = [np.random.randint(0, 256, size = (50, 360, 640, 3), dtype = 'uint8') for _ in range(9)]
        list_of_randomFrms = [BGRFrames(arr) for arr in random_arrays]

        myRst = FramesRaster(3, 3)

        stillRst, cascFrms = frame_cascade(list_of_randomFrms, myRst)
        DecapStream(cascFrms, 0, 100).show()

        stillRst.render().show()

        revRst, revFrms = reverse_cascade(stillRst, frames_per_element = 25, acceleration = 1.1, keep_which = [(1, 1)])
        DecapStream(revFrms, 0, 100).show()


    if TEST_DIAGRAMS:

        ### simple        
        diagram_text_array = [['factor 1', 'factor 2', 'factor 3'], [], ['output']]

        myDiag = Diagram(diagram_text_array)

        shape, arrow = myDiag[1, 0]
        shape.text_offset = 10
        shape.text_color = const.BGR['intense_red']
        myDiag.render().show()


        ### more stuff
        regressor_names = ['Stretch Distance', 'Returner uses Forehand', 'Return Depth', 'Serve Width', 
                           'Serve Speed', 'Serve Spin', 'Bounce Height']
        model_name = 'Propensity to Chip'
        myDiag = Diagram([regressor_names, [], [model_name]], resolution = 100, text_skip_after_how_many_symbols = 3,
                         shape_offset = 0.05, text_color = const.BGR['marine_blue'], render_height = 500)

        shape, arrow = myDiag[0, 1]
        remade_arrow = arrow.recolor(const.BGR['leafy_green']).resize(0.1)
        myDiag[0, 1] = shape, remade_arrow

        myDiag.render().show()

    if TEST_LEGEND:
        rows = [LegendRow('row 1', shape_color = const.BGR['intense_red']),
                LegendRow('row 2', shape_color = const.BGR['leafy_green'])]

        Legend(rows).render().show()

        rows = [LegendRow('row 1', shape_color = const.BGR['intense_red']),
                LegendRow(),
                LegendRow('row 2', shape_color = const.BGR['leafy_green'])]

        Legend(rows).render().show()

        Legend(rows, header_text = '').render().show()
