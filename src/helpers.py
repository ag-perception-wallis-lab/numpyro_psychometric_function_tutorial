"""
Collection of helper functions.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def get_top_directory() -> Path:
    """
    Find the path to the project's top-level directory.

    Depends on having the local package installed with `pip install -e`

    Returns:
    The path to the top-level directory
    """

    # use the location of the installed local package to figure out our absolute directory:
    tmp_path = Path(__file__).parent.parent.absolute()
    # assumes that we're running from a file in a subdirectory of the top dir.
    if (tmp_path / "tests").exists():
        # we've found the correct directory:
        return tmp_path
    else:
        raise ValueError(f"Couldn't find correct project directory; found {tmp_path}.")


def preprocess_data() -> None:
    """ """
    top_dir = get_top_directory()
    df = pd.concat(
        [pd.read_csv(f) for f in Path(top_dir / "data" / "raw_data").glob("*.csv")]
    )

    df.rename(
        columns={"correct_or_incorrect": "correct", "TrialNumber": "trial_number"},
        inplace=True,
    )
    df.loc[df["correct"] == "--", "correct"] = np.nan
    df["correct"] = df["correct"].astype(float)

    # note that target has +1 added, meaning response is [1, 2] whereas target is [2, 3].

    df["participant"] = pd.Categorical(df["participant"])
    df["participant"] = df["participant"].cat.rename_categories(
        ["participant_0", "participant_1"]
    )

    # drop na:
    df = df.dropna()

    df.to_csv(top_dir / "data" / "processed_data" / "demo_data_full.csv", index=False)
    return df
