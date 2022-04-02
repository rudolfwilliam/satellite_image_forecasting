from sentinelhub import SHConfig
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from get_coords import get_coords
from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
    DataCollection, bbox_to_dimensions, DownloadRequest

from sentinelhub import SHConfig

from sentinelhub import SHConfig

config = SHConfig()

config.instance_id = 'f9fd75fa-f692-4adb-b7cf-18a2e3941770'
config.sh_client_id = '4b40f784-9c31-4b2e-b938-34bba21b4b50'
config.sh_client_secret = 'MvM%sB^.6?i(#fWuqH!&|h7m@%uWMA(IBG0*P?&&'
config.save()