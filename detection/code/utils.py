# Miscellaneous utility function
#
# References
# ----------
# https://github.com/trzy/FasterRCNN//pytorch/FasterRCNN/utils.py

import torch

def no_grad(func):
    def wrapper_nograd(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper_nograd