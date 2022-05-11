import sys
from pathlib import Path
import pandas as pd
import yaml


def main():
    if len(sys.argv) != 3:
        print("ERROR; python store_metadata.py <df path (.csv)> <images dir>")

    df = pd.read_csv(sys.argv[1])
    df = df.apply(pd.to_numeric, errors='ignore')
    im_dir = Path(sys.argv[2])
    
    for i, f in df['FORMATTED'].items():
        f_path = next(im_dir.glob(f'**/{f}'))
        m_path = f_path.parent / 'metadata.yml'
        with open(m_path, mode='w') as stream:
            data = df.loc[i].to_dict()
            data = {k: v if isinstance(v, str) else v.item() for k, v in data.items()}  # dropping np dtype
            yaml.dump(data, stream)

if __name__ == '__main__':
    main()

