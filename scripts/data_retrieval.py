import earthnet as en

import sys
import os
from os.path import isdir, join

data_dir = sys.argv[1]
splits = sys.argv[2]

# create new data dir if necessary
if not isdir(join(os.getcwd(), data_dir)):
    os.mkdir(join(os.getcwd(), data_dir))

print(data_dir)
print(splits)
en.Downloader.get(data_dir, splits)
