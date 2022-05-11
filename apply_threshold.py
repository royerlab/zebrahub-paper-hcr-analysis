import click
import numpy as np
import pandas as pd
from pathlib import Path
from tifffile import imread, imwrite
from tqdm import tqdm


@click.command()
@click.option('--dataframe-path', '-d', type=click.Path(exists=True), help="Measurements dataframe path (*.csv)")
@click.option('--images-directory', '-i', type=click.Path(exists=True), help="Directory with images and labels.")
@click.option('--metadata-path', '-m', type=click.Path(exists=True), help="Image metadata containing columns CUTOFF_488 and CUTOFF_561")
def main(dataframe_path: str, images_directory: str, metadata_path: str) -> None:
    """
    Applies threshold given CUTOFFs to select the NM regions.
    """

    images_directory = Path(images_directory)

    df = pd.read_csv(dataframe_path)
    mt = pd.read_csv(metadata_path)
    mt['file'] = mt['FORMATTED'].apply(lambda x: x.replace('.tif', ''))
    mt = mt.set_index('file')[['CUTOFF_488', 'CUTOFF_561']]
    df = df.join(mt, on='file')
    df['selected'] = ((df['SOX2'] > df['CUTOFF_488']) & (df['TBXT'] > df['CUTOFF_561']))

    for f, group in tqdm(df.groupby('file'), "Selecting cells"):
        p = str(next(images_directory.glob(f'**/{f}_label.tif')))
        n_labels = group['label'].max() + 1
        mapping = np.zeros(n_labels, dtype=int)
        selected = group[group['selected']]['label'].values
        mapping[selected] = selected
        lb = imread(p)
        lb = mapping[lb]
        imwrite(p[:-4] + f'_thold.tif', lb)



if __name__ == '__main__':
    main()

