
# Installation

First, we recommend to create a new conda environment with

```bash
conda create -n HCR python=3.9
```

And activate it with `conda activate HCR`

Next install jupyter using anaconda, `conda install jupyterlab`.

Some packages are not in Python Package Index (pypi), so they have to be installed manually using github as follows:

 - [DEXP-DL](https://github.com/royerlab/dexp-dl):
    - Move to the directory where you wish to install DEXP-DL
    - `git clone https://github.com/royerlab/dexp-dl`
    - `cd dexp-dl`
    - `pip install -e .`

 - [sparse-deconv](https://github.com/JoOkuma/sparse-deconv-py) (OPTIONAL)
    - Move to the directory where you wish to install this package. We recommend to install in `~/Softwares` since this is where the code will look for it.
    - `git clone https://github.com/JoOkuma/sparse-deconv-py`
    - No installation is needed

And install remaining requirements:

```bash
pip install -r requirements
```

# Usage

Move to the directory of this repository and open jupyter-lab using

```bash
jupyter-lab .
```

- Download network weights from [here](https://drive.google.com/file/d/1vv9XpSOH9yEX1mOudFdoJnTR27Pkcvud/view?usp=sharing).
- Open your `.pynb` of choice.
- Update `IM_PATH`, `WEIGHTS_PATH`, and `SPARSE_DECONV_PATH` (optional) to reflect their location in your machine.
- Execute it.

