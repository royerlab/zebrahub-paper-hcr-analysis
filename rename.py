import click
import re
from pathlib import Path
import shutil
import pandas as pd
from tqdm import tqdm


def get_stage(formatted_path: Path) -> str:
    found = re.findall(r'(?<=hcr_)(bud|[0-9]*s)', str(formatted_path))
    return found[0]


def get_sample(formatted_path: Path) -> str:
    found = re.findall(r'(?<=[0-9]0x_)[0-9_]*(?=_)', str(formatted_path))
    return found[0]


def build_path(out_dir: Path, new_filename: Path) -> Path:
    stage_dir = out_dir / get_stage(new_filename)
    stage_dir.mkdir(exist_ok=True)
    sample_dir = stage_dir / get_sample(new_filename)
    sample_dir.mkdir(exist_ok=True)
    return sample_dir / new_filename


@click.command()
@click.option('--in-path', '-i', required=True, help='Existing source directory')
@click.option('--out-path', '-o', required=True, help='New formatted directory')
@click.option('--mapping-file', '-m', required=True, help='.csv file with mapping from `PREVIOUS`  to `FORMATTED` naming.')
def main(in_path: str, out_path: str, mapping_file: str):

    df = pd.read_csv(mapping_file, index_col=False)

    in_dir = Path(in_path)
    out_dir = Path(out_path)
    out_dir.mkdir(exist_ok=True)

    mapping = {
        prev: new
        for prev, new in zip(df['PREVIOUS'], df['FORMATTED'])
    }

    for prev, new in tqdm(mapping.items()):
        new_path = build_path(out_dir, new)
        prev_path = str(next(in_dir.glob(f'**/{prev}')))
        shutil.copy(str(prev_path), str(new_path))


if __name__ == '__main__':
    main()
