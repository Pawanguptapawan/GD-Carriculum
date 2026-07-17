import pandas as pd
import numpy as np
import glob
from pathlib import Path
URL='/Users/pagupta/Desktop/GD Projects/GD First Project/nfl-big-data-bowl-2026-prediction/train'
def load_data():
    folder_path=Path(URL)

    files=folder_path.glob("*.csv")

    if not files:
        raise FileNotFoundError('No data files found')
    dfs=[pd.read_csv(f) for f in  files]
    df=pd.concat(dfs,ignore_index=True)
    return df.sort_values(by=['game_id','play_id','nfl_id','frame_id']).reset_index(drop=True)

print(load_data())

