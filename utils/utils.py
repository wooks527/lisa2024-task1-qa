import monai
import numpy as np
import os
import random
import torch


def set_seed(seed: int) -> None:
    print(f"Set a random seed as {seed}.")

    os.environ["PYTHONHASHSEED"] = str(seed)  # os
    random.seed(seed)  # random
    np.random.seed(seed)  # numpy
    monai.utils.set_determinism(seed)  # monai

    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
