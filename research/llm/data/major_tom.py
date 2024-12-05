from io import BytesIO

import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from fsspec.parquet import open_parquet_file
from PIL import Image

PARQUET_FILE = "part_03900"  # parquet number
ROW_INDEX = 42  # row number (about 500 per parquet)

url = f"https://huggingface.co/datasets/Major-TOM/Core-S2L2A/resolve/main/images/{PARQUET_FILE}.parquet"
with open_parquet_file(url, columns=["thumbnail"]) as f:
    with pq.ParquetFile(f) as pf:
        first_row_group = pf.read_row_group(ROW_INDEX, columns=["thumbnail"])

stream = BytesIO(first_row_group["thumbnail"][0].as_py())
image = Image.open(stream)
plt.imshow(image)
plt.show()
