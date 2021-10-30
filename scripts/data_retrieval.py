import earthnet as en
import matplotlib.pyplot as plt

import numpy as np
import sys
import os
from os import path

data_dir = sys.argv[1]
splits = sys.argv[2]

# Create new data dir if necessary
if not path.isdir(os.path.join(os.getcwd(), data_dir)):
    os.mkdir(os.path.join(os.getcwd(), data_dir))

print(data_dir)
print(splits)
en.Downloader.get(data_dir, splits)

