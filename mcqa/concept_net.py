from pykeen.pipeline import pipeline
import matplotlib.pyplot as plt
import torch
from pykeen.datasets import Nations, Countries
from pykeen.pipeline import pipeline
from typing import List


result = pipeline(
    dataset='DB100K',
    model='PairRE',
    # Training configuration
    training_kwargs=dict(
        num_epochs=50,
        use_tqdm_batch=True,
    ),  
    # Runtime configuration
    random_seed=1235,
    device='gpu',
)

result.save_to_directory('graph')