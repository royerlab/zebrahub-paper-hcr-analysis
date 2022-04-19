
import sys
import numpy as np
import pandas as pd
import cupy as cp
import re

from typing import Sequence, Tuple, Optional
from pathlib import Path
from tifffile import imread, imwrite
from cucim.skimage.transform import downscale_local_mean
from skimage.measure import regionprops
from skimage.measure._regionprops import RegionProperties
from tqdm import tqdm

from dexp.processing.morphology import area_white_top_hat
from segmentation import segment_with_DL, segment_with_WS


DISPLAY = False
SAVE = True
CELL_CHANNEL = 0
CHANNELS = ['DAPI', 'TBXT', 'SOX2', 'OCT']
Z_SCALE = 2
AREA_OPENING_THOLD = 1e4
SUMMARY_FUN = np.sum


def find_image_paths(images_dir: Path) -> Sequence[Path]:
    errors = 'ERRORS\n'
    paths = []

    for im_path in tqdm(images_dir.glob('**/*.tif'), 'Checking images'):
        if any(suffix in str(im_path) for suffix in ('label', 'nobkg', 'measure')):
            continue
            
        image = imread(str(im_path))
        if image.ndim != 4:
            errors = errors + f" - Could not load {im_path}, expected 4 dimensions and found array of shape {image.shape}\n"
            continue

        paths.append(im_path)
    
    if errors != 'ERRORS\n':
        print(errors)
    
    stage_count = {}
    for im_path in paths:
        stage = get_stage(im_path)
        if stage not in stage_count:
            stage_count[stage] = 1
        else:
            stage_count[stage] += 1

    print(f'{len(paths)} images found')
    print('-------------------------')
    for stage, count in stage_count.items():
        print(f'{stage.ljust(5)}: {count} images')

    return paths


def get_stage(path: Path) -> str:
    return re.findall(r'(?<=\/)(bud|[0-9]+s)(?=\/)', str(path))[0]


def write_label(im_path: Path, label: np.ndarray) -> None:
    lb_path = str(im_path.with_suffix('')) + '_label.tif'
    imwrite(lb_path, label)


def process(image: np.ndarray, display: bool = False, im_path: Optional[Path] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    image = downscale_local_mean(cp.asarray(image), (1, Z_SCALE, 1, 1)).get()

    # removing background 
    no_bkg = np.stack([
        area_white_top_hat(image[i], 1e4, sampling=1, axis=0)
        for i in range(len(image))
    ])

    dapi_z_intensity = np.quantile(no_bkg[CELL_CHANNEL], q=0.999, axis=(1, 2))
    max_intensity = dapi_z_intensity.max()

    normalized = no_bkg / dapi_z_intensity[None, :, None, None]

    # normalized = ((max_intensity / dapi_z_intensity[None, :, None, None]) * no_bkg)
    # normalized = normalized.round().astype(no_bkg.dtype)

    if display:
        import napari
        viewer = napari.Viewer()
        viewer.add_image(image, name='original', channel_axis=0)
        viewer.add_image(no_bkg, name='processed', channel_axis=0)
        viewer.add_image(normalized, name='normalized', channel_axis=0)
        napari.run()
    
    labels = segment_with_WS(
        image[CELL_CHANNEL],
        display=display
    )
    props: Sequence[RegionProperties] = regionprops(labels, normalized.transpose((1, 2, 3, 0)))

    df = []

    for p in props:
        prop_feats = p.intensity_image[p.image]
        expressions = SUMMARY_FUN(prop_feats, axis=0)
        row = [
            p.label, *p.centroid, p.image.sum(), *expressions,
        ]
        df.append(row)

    df = pd.DataFrame(
        df,
        columns=['label', 'z', 'y', 'x', 'area'] + CHANNELS[:len(no_bkg)],
    )

    # saving data
    if im_path is not None:
        measurements = np.zeros((*no_bkg.shape[1:], no_bkg.shape[0]) , dtype=np.float32)
        for p in props:
            prop_feats = SUMMARY_FUN(p.intensity_image[p.image], axis=0)
            measurements[p.slice][p.image] = prop_feats # / (prop_feats[CELL_CHANNEL] + 1e-8)

        measurements = measurements.transpose((3, 0, 1, 2))
        imwrite(str(im_path.with_suffix('')) + '_nobkg.tif',  no_bkg)
        imwrite(str(im_path.with_suffix('')) + '_measure.tif',  measurements)

    return df, labels


if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print('ERROR, expected: python hcr_analysis.py <image directory>')
        sys.exit(-1)
    
    images_dir = Path(sys.argv[1])
    assert images_dir.exists()

    im_paths = find_image_paths(images_dir)
    dfs = []

    with tqdm(im_paths, desc='Processing') as pbar:
        for im_path in pbar:
            pbar.set_description(desc=f'{im_path.name}')
            image = imread(str(im_path))

            df, label = process(image, display=DISPLAY, im_path=im_path if SAVE else None)
            df['file'] = im_path.name.split('.', 1)[0]
            df['stage'] = get_stage(im_path)

            write_label(im_path, label)

            dfs.append(df)

            # updating at every iteration so I don't have to wait to it to finish
            df = pd.concat(dfs)
            df.to_csv('hcr_data.csv', index=False)
