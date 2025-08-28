from pathlib import Path
import pandas as pd

def Find_data_file(filename='MainData.xlsx', max_up=6):
    p = Path(__file__).resolve().parent
    for i in range(max_up):
        candidate = p / 'data' / filename
        if candidate.exists():
            return candidate
        p = p.parent
    # final attempt: project root style: try many parents for safety
    raise FileNotFoundError(f"Could not find {filename}. Checked up to {max_up} parent(s). "
                            f"CWD={Path.cwd()} ; script folder={Path(__file__).resolve().parent}")
