import nomic
import numpy as np
from datasets import load_dataset

dataset = load_dataset('ag_news')['train']

np.random.seed(0)
max_documents = 25000
subset_idxs = np.random.choice(
    len(dataset),
    size=max_documents,
    replace=False
).tolist()
documents = [dataset[i] for i in subset_idxs]

dataset = nomic.atlas.map_data(
    data=documents,
    indexed_field='text',
    identifier='News Dataset 25k',
    description='News Dataset 25k'
)

with dataset.wait_for_dataset_lock():
    dataset_map = dataset.maps[0]
    print(dataset_map.map_link)
    print(dataset.total_datums)

