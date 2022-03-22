from sentinelhub import SHConfig
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from get_coords import get_coords,get_limited_coords
from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
    DataCollection, bbox_to_dimensions, DownloadRequest

config = SHConfig()

if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")



def main():
    fn = "32UQC_2018-01-28_2018-11-23_5305_5433_4409_4537_82_162_68_148"
    coords = get_limited_coords(fn)
    data = get_data(coords)
    print(data.shape)
    plt.imshow(3.5*data[:,:,:3,22]/256)
    plt.show()

def get_data(coords, start = datetime.datetime(2020,1,28), end = datetime.datetime(2020,11,28), n_chunks = 60):
    #cords is a list with: lonmin, latmax, lonmax, latmin

    n_chunks += 1

    tdelta = (end - start) / n_chunks
    edges = [(start + i*tdelta).date().isoformat() for i in range(n_chunks)]
    slots = [(edges[i], edges[i+1]) for i in range(len(edges)-1)]

    resolution = 20
    bbox = BBox(bbox=coords, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution)

    evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04", "B08", "CLM"]
            }],
            output: {
                bands: 5
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02, sample.B08, sample.CLM];
    }
    """
    
    def get_true_color_request(time_interval):
        return SentinelHubRequest(
            evalscript=evalscript_true_color,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L1C,
                    time_interval=time_interval,
                    mosaicking_order='leastCC'
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.TIFF)
            ],
            bbox=bbox,
            size=size,
            config=config
        )

    # create a list of requests
    list_of_requests = [get_true_color_request(slot) for slot in slots]
    list_of_requests = [request.download_list[0] for request in list_of_requests]

    # download data with multiple threads
    data = SentinelHubDownloadClient(config=config).download(list_of_requests, max_threads=5)
    dataNew = np.asarray(data).transpose(1,2,3,0).astype(float)
    return dataNew

if __name__ == "__main__":
    main()