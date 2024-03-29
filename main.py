
import click
import numpy as np
import pandas as pd
import cupy as cp
import re

import yaml
from typing import Sequence, Tuple, Optional, Dict
from pathlib import Path
from tifffile import imread, imwrite
from cucim.skimage.transform import downscale_local_mean
from cucim.skimage.filters import gaussian
from skimage.measure import regionprops
from skimage.measure._regionprops import RegionProperties
from tqdm import tqdm
import napari

from dexp.processing.morphology import area_white_top_hat
from segmentation import segment_with_WS


DISPLAY = False
SAVE = True
CELL_CHANNEL = 0
CHANNELS = ['DAPI', 'SOX2', 'TBXT', 'OCT']
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


def correct_intensities(image: np.ndarray, metadata: Dict) -> np.ndarray:
    """Corrects image intensities according to exposure and laser power,
       it assumes axes are ordered according to wave-length.
    """
    assert CELL_CHANNEL == 0

    corrected = image.copy()

    corrected[1, ...] = corrected[1, ...] *\
            (metadata["LASERPOWER_405"] / metadata["LASERPOWER_488"]) *\
            (metadata["EXPOSURE_405"] / metadata["EXPOSURE_488"])

    corrected[2, ...] = corrected[2, ...] *\
            (metadata["LASERPOWER_405"] / metadata["LASERPOWER_561"]) *\
            (metadata["EXPOSURE_405"] / metadata["EXPOSURE_561"])

    # OCT-4 is not corrected
    return corrected


def process(image: np.ndarray, metadata: Dict, display: bool = False, im_path: Optional[Path] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    image = downscale_local_mean(cp.asarray(image), (1, Z_SCALE, 1, 1)).get()
    # image = correct_intensities(image, metadata)  # NOTE: not used, corrected on the tabular data.

    # removing background 
    no_bkg = np.stack([
        area_white_top_hat(image[i], 1e4, sampling=1, axis=0)
        for i in range(len(image))
    ])

    dapi = gaussian(cp.asarray(image[CELL_CHANNEL]), sigma=1).get()
    dapi_z_intensity = np.quantile(dapi, q=0.999, axis=(1, 2))

    # normalizing per z slice
    normalized = no_bkg / dapi_z_intensity[None, :, None, None]

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


@click.command('process')
@click.option('--images-dir', '-i', type=click.Path(exists=True, path_type=Path), help='cropped image directory', required=True)
@click.option('--out-path', '-o', type=click.Path(path_type=Path), help='.csv output path', required=True)
def process_cli(images_dir: Path, out_path: Path) -> None:

    im_paths = find_image_paths(images_dir)
    dfs = []

    with tqdm(im_paths, desc='Processing') as pbar:
        for im_path in pbar:
            pbar.set_description(desc=f'{im_path.name}')
            image = imread(str(im_path))

            with open(im_path.parent / 'metadata.yml') as f:
                metadata = yaml.safe_load(f)

            df, label = process(image, metadata=metadata, display=DISPLAY, im_path=im_path if SAVE else None)
            df['file'] = im_path.name.split('.', 1)[0]
            df['stage'] = get_stage(im_path)
            for k, v in metadata.items():
                if 'EXPOSURE' in k or 'LASERPOWER' in k:
                    df[k] = v

            write_label(im_path, label)

            dfs.append(df)

            # updating at every iteration so I don't have to wait to it to finish
            df = pd.concat(dfs)
            df.to_csv(out_path, index=False)


@click.command('figure')
@click.option('--image-dir', '-i', required=True, type=click.Path(exists=True, path_type=Path), help='input images (*_nobkg.etc, *_labels.tiff, etc.) path')
@click.option('--out-dir', '-o', required=True, type=click.Path(path_type=Path), help='output directory')
def figure_cli(image_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)

    image_path = str(next(image_dir.glob('*denoised.tif')))
    labels_path = image_path.replace('.tif', '_label.tif')
    nobkg_path = image_path.replace('.tif', '_nobkg.tif')
    measure_path = image_path.replace('.tif', '_measure.tif')

    with open(out_dir / 'LOG.txt', mode='w') as f:
        f.write(f'input directory: {image_dir}')

    v = napari.Viewer()
    v.window.resize(1080, 720)
    v.dims.ndisplay = 3

    def set_camera():
        v.camera.center = (25, 265, 235)
        v.camera.zoom = 1
        v.camera.angles = (-45, 55, 135)

    def screenshot(name):
        for i in range(3):
            v.layers[i].visible = True
            v.screenshot(out_dir / f'{name}_ch{i}.png')
            v.layers[i].visible = False
 
    v.add_image(imread(image_path), channel_axis=0, scale=(.5, 1, 1), visible=False)
    set_camera()
    screenshot('image')
    v.layers.clear()

    v.add_image(imread(nobkg_path), channel_axis=0, visible=False)
    set_camera()
    screenshot('nobkg')
    v.layers.clear()

    v.add_image(imread(measure_path), channel_axis=0, visible=False)
    set_camera()
    screenshot('measure')
    v.layers.clear()

    v.add_labels(imread(labels_path))
    set_camera()
    v.screenshot(out_dir / 'labels.png')


@click.group()
def main():
    pass


main.add_command(process_cli)
main.add_command(figure_cli)

if __name__ == '__main__':
    main()
