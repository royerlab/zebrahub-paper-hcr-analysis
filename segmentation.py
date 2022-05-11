
import numpy as np
import cupy as cp

from dexp.processing.morphology import area_closing

from cucim.skimage import morphology as morph
from cucim.skimage.filters import threshold_otsu
from cupyx.scipy.ndimage import median_filter
from skimage import segmentation
from pyift import shortestpath as sp


### ws parameters ###
AREA_THOLD = 1e4
WS_THOLD = 1


def segment_with_WS(image: np.ndarray, display: bool = False) -> np.ndarray:
    opened = morph.opening(cp.asarray(image),  morph.ball(np.sqrt(2)))
    closed = cp.asarray(area_closing(opened.get(), AREA_THOLD, sampling=1, axis=0))
    closed = median_filter(closed, footprint=cp.ones((3, 1, 1), dtype=bool))

    thold = threshold_otsu(closed)
    detection = (closed > thold).get()

    basins = opened / np.quantile(opened, 0.999)
    basins = basins.max() - basins
    basins = np.sqrt(basins)
    _, labels = sp.watershed_from_minima(basins.get(), detection, H_minima=0.05, compactness=0.005)
    labels[labels < 0] = 0
    labels, _, _ = segmentation.relabel_sequential(labels)

    if display:
        import napari
        viewer = napari.Viewer()

        viewer.add_image(opened.get())
        viewer.add_image(closed.get())
        viewer.add_labels(detection)
        viewer.add_labels(labels)

        napari.run()
    
    return labels
