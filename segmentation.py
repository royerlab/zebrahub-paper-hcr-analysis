
import numpy as np
import torch as th
import torch.nn.functional as F
import cupy as cp
import higra as hg

from dexp_dl.models import hrnet
from dexp_dl.inference import ModelInference
from dexp_dl.postprocessing import hierarchy
from dexp.processing.morphology import area_closing
from dexp.processing.morphology.utils import get_3d_image_graph

from cucim.skimage import morphology as morph
from cucim.skimage.filters import threshold_otsu
from cupyx.scipy.ndimage import median_filter
from skimage import segmentation
from pyift import shortestpath as sp


### deep learning params ###
NETWORK_WEIGHTS_PATH = 'network_weights.ckpt'
NETWORK_N_CLASSES = 3
DEVICE = 0
QUANTILE = 0.999
PRED_THOLD = 0.25
SEGM_THOLD = 1
WS_HIERARCHY = hg.watershed_hierarchy_by_area

### ws parameters ###
AREA_THOLD = 1e4
WS_THOLD = 1

def in_transform(image):
    return th.Tensor(image).unsqueeze_(0).half()


def out_transform(image):
    return th.sigmoid(F.interpolate(image, scale_factor=2, mode='trilinear', align_corners=True))


th.cuda.device(DEVICE)

net = hrnet.hrnet_w18_small_v2(
    pretrained=False, in_chans=1, num_classes=NETWORK_N_CLASSES, image_ndim=3
)

model = ModelInference(
    net,
    transforms=in_transform,
    after_transforms=out_transform,
    tile=(48, 96, 96), num_outputs=NETWORK_N_CLASSES,
)

model.load_weights(NETWORK_WEIGHTS_PATH)


def segment_with_DL(image: np.ndarray, display: bool = False) -> np.ndarray:
    normalized = image - image.min()
    quantile = np.quantile(normalized, QUANTILE)
    normalized = np.clip(normalized / quantile, 0, 1)

    with th.cuda.amp.autocast():
        pred = model(normalized)
        
    th.cuda.empty_cache()  

    hiers = hierarchy.create_hierarchies(
        pred[0] > PRED_THOLD,
        pred[1],
        hierarchy_fun=WS_HIERARCHY,
        cache=True,
        min_area=10,
        min_frontier=0,
    )
        
    for h in hiers:
        h.cut_threshold = SEGM_THOLD

    labels = hierarchy.to_labels(hiers, pred[0].shape)
    if display:
        import napari
        viewer = napari.Viewer()
        viewer.add_image(image, name='input')
        viewer.add_image(pred[0], name='detection')
        viewer.add_image(pred[1], name='boundary')
        viewer.add_labels(labels, name='segments').contour = 1
        napari.run()

    return labels


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
