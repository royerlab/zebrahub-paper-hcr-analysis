[![DOI](https://zenodo.org/badge/427429472.svg)](https://zenodo.org/doi/10.5281/zenodo.13761254)

# HCR Analysis files

## Installation

First, we recommend to create a new conda environment with

```bash
conda create -n HCR python=3.9
```

And activate it with `conda activate HCR` and install requirements with:

```bash
pip install -r requirements
```

## Files Documentation

- `main.py`: This is the principal file that extracts the main statistics from the HCR images. It has two sub commands:

   - `process`: Process our standardized directory structure writing an output `.csv` file and saving auxiliary images at the input images directory.

   - `figure`: Create images of intermediate steps from the `process` function.

- `apply_threshold.py`: Applies a threshold from the metadata file to the TBXT and SOX2 measurements selecting the cells which express both and save it to the input images directory.

- `rename.py`: Renames and reorganizes the files to our standard format using the metadata file.

- `store_metadata.py`: Auxiliary file to map metadata from table to the standardized directory path.

- `max_project_figures.py`: Creates the images of the maximum intensity projections overlaid with the selected cells.

- `count_plot.Rmd`: Generates the count barplot given the processed measurements file and the threshold annotation file. File must be edited to set input paths.
