"""Sample faces, for scoring an export against eager.

The model only ever sees a tight crop of one face, so a general photo set would
say nothing about the inputs it actually meets. CelebA's aligned crops have the
right framing: one centred face filling a 178x218 frame.

LFW would be the other obvious choice, but neither torchvision's downloader nor
the figshare mirror still serves it.
"""

import io
import random

import torch

#: Faces the rejected int8 calibration consumed; scoring skips past them so the
#: two sets stay disjoint. See the int8 section of NOTES.md.
CALIBRATION_SIZE = 100

_REPO = "tpremoli/CelebA-attrs"
_FILE = "data/test-00000-of-00001.parquet"


def images(count: int = CALIBRATION_SIZE, skip: int = 0):
    """Yield `count` PIL faces from the cached CelebA shard, past the first `skip`.

    The order is seeded, so `skip=CALIBRATION_SIZE` reproduces the faces every
    export in NOTES.md was scored on.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    path = hf_hub_download(_REPO, _FILE, repo_type="dataset")
    # Read row groups until there are enough rows, rather than pulling the
    # whole 140 MB shard into memory for a hundred-odd faces.
    reader = pq.ParquetFile(path)
    rows = []
    for group in range(reader.num_row_groups):
        if len(rows) >= skip + count:
            break
        rows += reader.read_row_group(group, columns=["image"]).column("image").to_pylist()
    random.seed(0)
    for row in random.sample(rows, k=skip + count)[skip:]:
        yield Image.open(io.BytesIO(row["bytes"])).convert("RGB")
