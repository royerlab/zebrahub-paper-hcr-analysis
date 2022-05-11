import click
import napari
import numpy as np
from pathlib import Path
from tifffile import imread
from skimage.segmentation import find_boundaries
import skimage.morphology as morph
from tqdm import tqdm


def make_figure(im_path: Path, figs_dir: Path) -> None:
    v = napari.Viewer()
    v.window.resize(960, 960)

    measures = imread(im_path)
    measures = measures.max(axis=1)
    measures_layers = v.add_image(measures, channel_axis=0)
    for layer in measures_layers:
        upper_limit = np.quantile(layer.data, 0.999)
        layer.contrast_limits = (layer.data.min(), upper_limit)
    
    # hiding OCT layer
    for layer in measures_layers[3:]:
        layer.visible = False

    labels_path = list(im_path.parent.glob('*_thold.tif'))
    assert len(labels_path) > 0

    if len(labels_path) > 1:
        print('Multiple thresholds found. Using a single one.')

    label = imread(labels_path[0])
    label = label.max(axis=0)
    contour = find_boundaries(label, mode='inner').astype(np.uint8)
    outer = morph.binary_dilation(contour, selem=morph.disk(1))
    outer[contour > 0] = 0
    contour[outer > 0] = 2
    
    contour[-1:,-50:] = 1  # reference line

    cmap = {
        0: "transparent",
        1: "white",
        2: 'black',
    }
    label_layer = v.add_labels(
        contour,
        color=cmap,
        opacity=1.0,
    )

    v.reset_view()

    fig_name = im_path.name.split('measure', 1)[0] + 'threshold_figure.png'
    v.screenshot(figs_dir / fig_name)

    # napari.run()


@click.command()
@click.argument("data_dir", nargs=1)
@click.option('--figs-dir', '-f', type=click.Path(path_type=Path), default=Path('hcr-figures'))
def main(data_dir: str, figs_dir: Path) -> None:

    data_dir = Path(data_dir)
    figs_dir.mkdir(exist_ok=True)

    for im_path in tqdm(list(data_dir.glob('**/*_measure.tif'))):
        make_figure(im_path, figs_dir)


if __name__ == '__main__':
    main()
