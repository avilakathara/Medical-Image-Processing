import sys
import napari
import numpy as np
from PIL import Image
# conda install -c conda-forge pyift
# from pyift.livewire import LiveWire
from skimage import color
from magicgui import magicgui
from qtpy.QtWidgets import QDoubleSpinBox
from qtpy.QtWidgets import QSlider
from pathlib import Path
from matplotlib.patches import Rectangle
from napari.settings import SETTINGS
import numpy as np
import _pyift
import collections
from typing import Union, Sequence, Optional
import warnings

# pip install dijkstra
from dijkstra import Graph, DijkstraSPF

# conda install -c conda-forge opencv
import cv2 as cv

from matplotlib import pyplot as plt



class LiveWire:
    current: Optional[np.ndarray]
    saliency: Optional[np.ndarray]

    def __init__(self, image: np.ndarray, arc_fun: str = 'exp', saliency: Optional[np.ndarray] = None, **kwargs):
        """
        Live-wire object to iteratively compute the optimum-paths [1]_ between user selected points.

        Parameters
        ----------
        image: array_like
            Array where the first two dimensions are the image domain, the third and optional are its features.
        arc_fun: {'exp'}, default='exp'
            Optimum-path arc-weight function.
        saliency: array_like, optional
            Array with the same dimension as the image domain containing the foreground saliency.
        kwargs: float, optional
            Key word arguments for arc-weight function parameters.

        Attributes
        ----------
        arc_fun: {'exp'}, default='exp'
            Optimum-path arc-weight function.
        image: array_like
            Array where the first two dimensions are the image domain, the third and optional are its features.
        saliency: array_like, optional
            Array with additional features, usually object saliency. Must have the same domain as `image`.
        costs: array_like
            Array containing the optimum-path cost.
        preds: array_like
            Array containing the predecessor map to recover the optimal contour.
        labels: array_like
            Array indicating optimum-path nodes
        size: tuple
            Tuple containing the image domain.
        sigma: float
            Image features parameter.
        gamma: float
            Saliency features parameter.
        source: int
            Current path source node index (flattened array), -1 if inactive.
        destiny: int
            Current path destiny node index (flattened array), -1 if inactive.
        start: int
            Current contour starting node index, -1 if inactive.
        current: array_like
            Active optimum-path, before confirmation.
        paths: dict
            Ordered dictionary of paths, key: path source, value: path sequence.

        Examples
        --------

        # >>> import numpy as np
        # >>> from pyift.livewire import LiveWire
        # >>>
        # >>> image = np.array([[8, 1, 0, 2, 0],
        # >>>                   [5, 7, 2, 0, 1],
        # >>>                   [6, 7, 6, 1, 0],
        # >>>                   [6, 8, 7, 0, 3],
        # >>>                   [6, 7, 8, 8, 9]])
        # >>>
        # >>> lw = LiveWire(image, sigma=1.0)
        # >>> lw.select((0, 0))
        # >>> lw.confirm()
        # >>> lw.select((4, 4))
        # >>> lw.confirm()
        # >>> lw.contour

        References
        ----------
        .. [1] Falcão, Alexandre X., et al. "User-steered image segmentation paradigms:
               Live wire and live lane." Graphical models and image processing 60.4 (1998): 233-260.

        """
        if not isinstance(image, np.ndarray):
            raise TypeError('`image` must be a `ndarray`.')

        if image.ndim == 2:
            image = np.expand_dims(image, 2)

        if image.ndim != 3:
            raise ValueError('`image` must 2 or 3-dimensional array.')

        if saliency is not None:
            if not isinstance(saliency, np.ndarray):
                raise TypeError('`saliency` must be a `ndarray`.')

            if saliency.ndim == 2:
                saliency = np.expand_dims(saliency, 2)

            if saliency.ndim != 3:
                raise ValueError('`saliency` must 2 or 3-dimensional array.')

            if saliency.shape[:2] != image.shape[:2]:
                raise ValueError('`saliency` and `image` 0,1-dimensions must match.')

            self.saliency = np.ascontiguousarray(saliency.astype(float))

        arc_functions = ('exp', 'exp-saliency')
        if arc_fun.lower() not in arc_functions:
            raise ValueError('Arc-weight function not found, must include {}'.format(arc_functions))

        self.arc_fun = arc_fun.lower()

        if self.arc_fun.startswith('exp'):
            sigma = 1.0
            if 'sigma' not in kwargs:
                warnings.warn('`sigma` not provided, using default, %f' % sigma, Warning)
            self.sigma = kwargs.pop('sigma', sigma)

        if self.arc_fun == 'exp-saliency':
            if saliency is None:
                raise TypeError('`saliency` must be provided with `exp-saliency` arc-weight.')
            gamma = 1.0
            if 'gamma' not in kwargs:
                warnings.warn('`gamma` not provided, using default %f' % gamma, Warning)
            self.gamma = kwargs.pop('gamma', gamma)

        self.size = image.shape[:2]
        self.image = np.ascontiguousarray(image.astype(float))
        self.costs = np.full(self.size, np.finfo('d').max, dtype=float)
        self.preds = np.full(self.size, -1, dtype=int)
        self.labels = np.zeros(self.size, dtype=bool)

        self.float_destiny = -1.0
        self.float_source = -1.0
        self.source = -1
        self.destiny = -1
        self.float_start = -1.0
        self.start = -1
        self.paths = collections.OrderedDict()
        self.current = None
        self.laplacian = abs(cv.Laplacian(image,cv.CV_32F).astype(int))
        self.max_gradient = np.max(self.laplacian)
        self.distance_to_start = np.zeros(image.shape)
        self.chosen_slice = 0
        self.straight = False

    def _opt_path(self, src: int, dst: int) -> Optional[np.ndarray]:

        """
        Compute optimum-path from source to destiny.

        Parameters
        ----------
        src : int
            Source index.
        dst : int
            Destiny index.

        Returns
        -------
        array_like
            Array of flattened indices.
        """
        if self.arc_fun == 'exp':
            # gaat mis!!!!!!!!!
            # path = _pyift.livewire_path(self.image, self.costs, self.preds, self.labels,
            #                             self.arc_fun, self.sigma, src, dst)
            path = self.my_optimal_path(src, dst)
        elif self.arc_fun == 'exp-saliency':
            path = _pyift.livewire_path(self.image, self.saliency, self.costs, self.preds, self.labels,
                                        self.arc_fun, self.sigma, self.gamma, src, dst)
        else:
            raise NotImplementedError
        return path

    def _assert_valid(self, y: int, x: int) -> None:
        """
        Asserts coordinates belong in the image domain.
        """
        if not (0 <= y < self.size[0] and 0 <= x < self.size[1]):
            print(y,x)
            raise ValueError('Coordinates out of image boundary, {}'.format(self.size))

    def select(self, position: Union[Sequence[int], int]) -> None:
        """
        Selects next position to compute optimum-path to, or initial position.

        Parameters
        ----------
        position: Sequence[int, int], int
            Index or coordinate (y, x) in the image domain
        """
        float_pos = position
        if isinstance(position, (list, tuple, np.ndarray)):
            y, x = round(position[0]), round(position[1])
            self._assert_valid(y, x)
            position = int(y * self.size[1] + x)

        if not isinstance(position, int):
            raise TypeError('`position` must be a integer, tuple or list.')

        if self.source != -1:
            # hier gaat het mis!!!!!!!!!!!!!!!!!
            self.cancel()
            # self.current = self.my_optimal_path(self.float_source,float_pos)
            self.current = self._opt_path(self.float_source, float_pos)
        else:
            for i in range(len(image)):
                for j in range(len(image[0])):
                    self.distance_to_start[i,j] = np.sqrt((((float_pos[0]-i)**2) + (float_pos[1]-j)**2))
            self.start = position
            self.float_start = float_pos
        self.destiny = position  # must be after cancel
        self.float_destiny = float_pos

    def cancel(self) -> None:
        """
        Cancel current unconfirmed path.
        """
        if self.current is not None and self.current.size:
            # reset path
            self.labels.flat[self.current] = False
            self.costs.flat[self.current] = np.finfo('d').max
            # reset path end
            self.labels.flat[self.destiny] = False
            self.costs.flat[self.destiny] = np.finfo('d').max

    def confirm(self) -> None:
        """
        Confirms current path and sets it as new path source.
        """
        if self.source != -1:
            self.paths[self.source] = self.current
        self.source = self.destiny
        self.float_source = self.float_destiny
        self.current = None

    def close(self) -> None:
        """
        Connects the current path to the initial coordinate, closing the live-wire contour. Result must be confirmed.
        """

        with open('../data.txt', 'a') as f:
            f.write("\n")
            final_points = []
            for path in self.paths.values():
                for point in path:
                    coordinates = flat_to_indices(point,self.size[1])
                    final_points.append([self.chosen_slice,coordinates[0],coordinates[1]])

            f.write(str(final_points))


        # if len(self.paths) == 0:
        #     raise ValueError('Path must be confirmed before closing contour')
        self.cancel()
        self.costs.flat[self.start] = np.finfo('d').max
        self.select(flat_to_indices(self.start,self.size[1]))
        self.confirm()
        self.start = -1
        self.float_start = -1.0
        self.source = -1
        self.float_source = -1.0
        self.destiny = -1
        self.float_destiny = -1.0

    @property
    def contour(self) -> np.ndarray:
        """
        Returns
        -------
        array_like
            Optimum-path contour.
        """
        return self.labels

    def draw_line(self,mat, x0, y0, x1, y1, inplace=False):
        if not (0 <= x0 < mat.shape[0] and 0 <= x1 < mat.shape[0] and
                0 <= y0 < mat.shape[1] and 0 <= y1 < mat.shape[1]):
            raise ValueError('Invalid coordinates.')
        if not inplace:
            mat = mat.copy()
        # if (x0, y0) == (x1, y1):
        #     mat[x0, y0] = 2
        #     return mat if not inplace else None
        # Swap axes if Y slope is smaller than X slope
        transpose = abs(x1 - x0) < abs(y1 - y0)
        if transpose:
            mat = mat.T
            x0, y0, x1, y1 = y0, x0, y1, x1
        # Swap line direction to go left-to-right if necessary
        if x0 > x1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        # Write line ends
        mat[x0, y0] = 2
        mat[x1, y1] = 2
        # Compute intermediate coordinates using line equation
        if x1==x0:
            y = np.arange(y0 + 1, y1)
            x = np.round(np.ones(len(y)) * x1).astype(y.dtype)
        else:
            x = np.arange(x0 + 1, x1)
            y = np.round(((y1 - y0) / (x1 - x0)) * (x - x0) + y0).astype(x.dtype)
        path = []
        for i in range(len(x)):
            if transpose:
                path.append(flatten_indices([y[i],x[i]],self.size[1]))
            else:
                path.append(flatten_indices([x[i],y[i]],self.size[1]))
        return path

    def my_optimal_path_straight(self,s, d):
        src = s
        dst = d

        return self.draw_line(np.zeros(self.size),round(src[0]),round(src[1]),round(dst[0]),round(dst[1]))

    def create_graph(self,src,dst,size,image,box):

        def cost(a,b):
            c = 0
            for i in range(len(b)):
                for j in range(len(b[0])):
                    c += 1-self.laplacian[i,j]/self.max_gradient
            return c

        includes_src = False
        includes_dst = False
        graph = Graph()
        for point in box:
            indices = flat_to_indices(point,self.size[1])
            x,y = int(indices[1]-0.5*size),int(indices[0]-0.5*size)
            if x < 1 or y < 1 or x >= image.shape[1] - size or y >= image.shape[0] - size:
                continue
            if point == src:
                includes_src = True
            if point == dst:
                includes_dst = True

            window = image[y:y+size,x:x+size]
            to = flatten_indices([indices[0],indices[1]+1],self.size[1])
            graph.add_edge(point,to,cost(window,image[y:y+size,x+1:x+1+size]))

            to = flatten_indices([indices[0],indices[1]-1],self.size[1])
            graph.add_edge(point,to,cost(window,image[y:y+size,x-1:x-1+size]))

            to = flatten_indices([indices[0]+1,indices[1]],self.size[1])
            graph.add_edge(point,to,cost(window,image[y+1:y+1+size,x:x+size]))

            to = flatten_indices([indices[0]-1,indices[1]],self.size[1])
            graph.add_edge(point,to,cost(window,image[y-1:y-1+size,x:x+size]))

        if not includes_dst and not includes_src:
            return self.my_optimal_path_straight(flat_to_indices(src,self.size[1]),flat_to_indices(dst,self.size[1]))

        dijkstra = DijkstraSPF(graph, src)
        return dijkstra.get_path(dst)










    def my_optimal_path(self,s, d):
        path = []
        src = s
        dst = d
        size = 10

        if self.straight or src[0] < size or src[1] < size or dst[0] < size or dst[1] < size or src[0] > image.shape[0]-size or src[1] > image.shape[1]- size or dst[0] > image.shape[0]- size or dst[1] > image.shape[1]- size:
            path = self.my_optimal_path_straight(src,dst)
            for point in path:
                self.labels.flat[point] = True
            return np.array(path)

        # miss nog eerst morphology of blur/ander filter?
        offset_x, offset_y = 6,6
        dist = np.sqrt(((src[0]-dst[0])**2) + (src[1]-dst[1])**2)
        center = [(src[0]+dst[0])/2,(src[1]+dst[1])/2]

        if src[1] == dst[1]:
            angle = 90
        elif src[0] == dst[0]:
            angle = 0
        else:
            angle = np.degrees(np.arctan(abs(src[0]-dst[0])/abs(src[1]-dst[1])))
        if (src[0] < dst[0] and src[1] > dst[1]) or (src[0] > dst[0] and src[1] < dst[1]):
            angle = 360-angle

        x,y = center[1]-0.5*dist-offset_x,center[0]-offset_y
        width = 2*offset_x+dist
        height = 2*offset_y
        rect = Rectangle((x,y), width,height, angle=angle, rotation_point='center')
        corners = rect.get_corners()

        points = []
        for p in corners:
            points.append([p[0],p[1]])

        inside_rect = []
        mask = np.zeros((image.shape[0],image.shape[1],3))
        cv.fillPoly(mask,pts=[np.array(points,dtype=np.int32)],color=(0, 255, 0))
        for i in range(len(mask)):
            for j in range(len(mask[0])):
                if sum(mask[i][j]) > 0:
                    inside_rect.append(flatten_indices([i,j],self.size[1]))


        path = self.create_graph(flatten_indices(src,self.size[1]),flatten_indices(dst,self.size[1]),size,self.image,inside_rect)
        for point in path:
            self.labels.flat[point] = True
        return np.array(path)



def flat_to_indices(index,size_x):
    y = int(index/size_x)
    x = index - y*size_x
    return [y,x]

def flatten_indices(indices,size_x):
    x = round(indices[1])
    y = round(indices[0])
    return round(y * size_x + x)






# SETTINGS.reset()
SETTINGS.experimental.async_ = True

patient_dest_dir = Path("/Users/Bram/Documents/RP/miccai_data_numpy/part1/0522c0002")
images = np.load(patient_dest_dir.joinpath("img.npy"))
image = images[65]
default_sigma = 5.0


with open('../data.txt', 'w') as f:
    f.write("")


viewer = napari.view_image(images, rgb=False)
livewire = LiveWire(image, sigma=default_sigma)
livewire.chosen_slice = 65
livewire.straight = True
layer = viewer.add_labels(livewire.contour,
                          name='contour', opacity=1.0)

def valid(coords):
    return 0 <= round(coords[0]) < image.shape[0] and 0 <= round(coords[1]) < image.shape[1]

click_count = 0




@layer.mouse_move_callbacks.append
def mouse_move(layer, event):
    global image
    global livewire
    global click_count

    coords = event.position[1:]
    if livewire.chosen_slice != event.position[0]:
        click_count = 0
        print("new slice!")


        image = images[round(event.position[0])]
        livewire = LiveWire(image, sigma=default_sigma)
        livewire.chosen_slice = event.position[0]
        livewire.straight = True
        layer.data = livewire.contour

    if valid(coords):
        best_gradient = abs(livewire.laplacian[int(coords[0]),int(coords[1])])
        best_coords = coords
        snap_width = 0
        for i in range(int(max(0,coords[0]-snap_width)), int(1+min(image.shape[0]-1,coords[0]+snap_width))):
            for j in range(int(max(0,coords[1]-snap_width)), int(1+min(image.shape[1]-1,coords[1]+snap_width))):
                if abs(livewire.laplacian[i,j]) > best_gradient:
                    best_gradient = abs(livewire.laplacian[i,j])
                    best_coords = [i,j]
        if valid(best_coords):
            coords = best_coords

        livewire.select(coords)
        layer.data = livewire.contour



@layer.mouse_drag_callbacks.append
def mouse_click(layer, event):
    global click_count
    livewire.confirm()
    click_count += 1
    print(click_count)
    if click_count > 2 and livewire.distance_to_start[int(event.position[1])][int(event.position[2])] < 5:
        livewire.select(flat_to_indices(livewire.start,livewire.size[1]))
        layer.data = livewire.contour
        livewire.close()
        click_count = 0

napari.run()



