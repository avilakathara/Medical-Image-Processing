# import sys
import napari
# import numpy as np
# from PIL import Image
# conda install -c conda-forge pyift
# from pyift.livewire import LiveWire
# from skimage import color
# from magicgui import magicgui
# from qtpy.QtWidgets import QDoubleSpinBox
# from qtpy.QtWidgets import QSlider
from pathlib import Path
# from matplotlib.patches import Rectangle
# from napari.settings import SETTINGS
import numpy as np
import _pyift
import collections
from typing import Union, Sequence, Optional
import warnings
# from skimage.morphology import binary_dilation
# from skimage.morphology import binary_erosion
# pip install dijkstra
from dijkstra import Graph, DijkstraSPF

# conda install -c conda-forge opencv
import cv2 as cv

# from matplotlib import pyplot as plt


class LiveWire2:
    def __init__(self, image: np.ndarray, arc_fun: str = 'exp', saliency: Optional[np.ndarray] = None, **kwargs):
        self.size = image.shape[1:]
        self.image = np.ascontiguousarray(image.astype(float))
        self.open = True
        # self.seed_points = []
        self.source = -1
        self.start = -1
        # self.current_path = []
        self.drawing = np.zeros(image.shape, dtype=bool)
        self.current_drawing = np.zeros(image.shape, dtype=bool)

        self.position = -1
        # self.laplacian_zc, self.laplacian = get_laplacians(image)
        # self.max_gradient = np.max(self.laplacian)
        # self.distance_to_start = np.zeros(image.shape)
        self.laplacian = None
        self.max_gradient = None
        self.current_slice = -1

        self.dijkstra = None
        self.dijkstra_nodes = []


    def reset(self):
        if self.open:
            print('delete slice ',self.current_slice)
            self.drawing[self.current_slice] = np.zeros(self.size,dtype=bool)
        self.open = False
        self.source = -1
        self.start = -1
        self.position = -1

    def select(self, position):
        self.position = position
        self.current_drawing = np.zeros(self.image.shape, dtype=bool)

        if self.source == -1:
            return

        src = flat_to_indices(self.source,self.size[1])
        dst = flat_to_indices(position,self.size[1])
        path = []
        if position in self.dijkstra_nodes:
            path = self.dijkstra.get_path(position)

        for index in path:
            pixel = flat_to_indices(index,self.size[1])
            self.current_drawing[self.current_slice,round(pixel[0]),round(pixel[1])] = True


    def clicked(self):
        if self.source != -1:
            if self.position == self.start:
                self.open = False
                self.reset()
            self.drawing[self.current_drawing] = True

        else:
            self.open = True
            self.start = self.position

        self.source = self.position
        self.dijkstra, self.dijkstra_nodes = create_dijkstra(self.size,self.source,self.laplacian,self.max_gradient)

def create_dijkstra(size,source,laplacian,max_gradient):

    def cost(p,q):
        l = 1-laplacian.flat[q]/max_gradient
        return l

    indices = flat_to_indices(source,size[1])
    x_source, y_source = indices[1],indices[0]
    graph = Graph()
    window_size = 50
    box = []
    nodes = []
    for i in range(max(0,y_source-window_size),min(size[0]-1,y_source+window_size)):
        for j in range(max(0,x_source-window_size),min(size[1]-1,x_source+window_size)):
            box.append([i,j])


    for point in box:
        point_flat = flatten_indices(point,size[1])
        if point[1] < size[1]-1:
            to = flatten_indices([point[0],point[1]+1],size[1])
            nodes.append(to)
            graph.add_edge(point_flat,to,cost(source,to))
        if point[1] > 0:
            to = flatten_indices([point[0],point[1]-1],size[1])
            nodes.append(to)

        graph.add_edge(point_flat,to,cost(source,to))
        if point[0] < size[0]-1:
            to = flatten_indices([point[0]+1,point[1]],size[1])
            nodes.append(to)

        graph.add_edge(point_flat,to,cost(source,to))
        if point[0] > 0:
            to = flatten_indices([point[0]-1,point[1]],size[1])
            nodes.append(to)

        graph.add_edge(point_flat,to,cost(source,to))

    dijkstra = DijkstraSPF(graph, source)
    return dijkstra, nodes


def flat_to_indices(index,size_x):
    y = int(index/size_x)
    x = index - y*size_x
    return [y,x]

def flatten_indices(indices,size_x):
    x = round(indices[1])
    y = round(indices[0])
    return round(y * size_x + x)


def automatic_contours(ground_truth):
    result = np.zeros(ground_truth.shape,dtype=bool)
    target = int(len(ground_truth)/2)
    result[target] = create_contour(ground_truth[target])
    return result

def create_contour(ground_truth):
    if np.count_nonzero(ground_truth) == 0:
        return ground_truth
    result = np.zeros(ground_truth.shape,dtype=bool)
    im = ground_truth.astype(np.uint8)
    contours, hierarchy = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        for point in contour:
            result[point[0][1], point[0][0]] = True
    return result



