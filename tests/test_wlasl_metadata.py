from pathlib import Path

import pandas as pd
import pytest


def test_wlasl_metadata_files_exist() -> None:
    path = Path("data/raw/wlasl/metadata.csv")
    if not path.exists():
        pytest.skip("WLASL subset has not been downloaded")
    metadata = pd.read_csv(path)
    required = {"video_path", "label", "signer_id", "original_split", "duration_sec", "fps", "width", "height"}
    assert required <= set(metadata.columns)
    for relative_path in metadata["video_path"]:
        assert Path(relative_path).exists()
