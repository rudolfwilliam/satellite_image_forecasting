

import os
import json
from pyproj import Transformer
from earthnet.coords_dict import COORDS
def main():
    file_name = "32UQC_2018-01-28_2018-11-23_5305_5433_3257_3385_82_162_50_130.npz"
    print(get_coords_from_cube(file_name))
    #51.308834, 13.230445

def get_coords_from_cube(cubename: str, return_meso: bool = False):
    """

    Get the coordinates for a Cube in Lon-Lat-Grid.

    Args:
        cubename (str): cubename (has format tile_startyear_startmonth_startday_endyear_endmonth_endday_hrxmin_hrxmax_hrymin_hrymax_mesoxmin_mesoxmax_mesoymin_mesoymax.npz)
        return_meso (bool, optional): If True returns also the coordinates for the Meso-scale variables in the cube. Defaults to False.

    Returns:
        tuple: Min-Lon, Min-Lat, Max-Lon, Max-Lat or Min-Lon-HR, Min-Lat-HR, Max-Lon-HR, Max-Lat-HR, Min-Lon-Meso, Min-Lat-Meso, Max-Lon-Meso, Max-Lat-Meso
    """    
    cubetile,_, _,  hr_y_min, hr_y_max,hr_x_min, hr_x_max, meso_x_min, meso_x_max, meso_y_min, meso_y_max = os.path.splitext(cubename)[0].split("_")

    tile = COORDS[cubetile]

    transformer = Transformer.from_crs(tile["EPSG"], 4326, always_xy = True)

    tile_x_min, tile_y_max = transformer.transform(tile["MinLon"],tile["MaxLat"], direction = "INVERSE")

    a = 20

    cube_x_min = tile_x_min + a * float(hr_x_min)
    cube_x_max = tile_x_min + a * float(hr_x_max)
    cube_y_min = tile_y_max - a * float(hr_y_min)
    cube_y_max = tile_y_max - a * float(hr_y_max)
    
    cube_lon_min, cube_lat_min = transformer.transform(cube_x_min, cube_y_max)
    cube_lon_max, cube_lat_max = transformer.transform(cube_x_max, cube_y_min)

    print("{0}, {1}".format(cube_lat_min,cube_lon_min))
    cube_lat_min
    
    return cube_lon_min, cube_lat_min, cube_lon_max, cube_lat_max

def get_coords_from_tile(tilename: str):
    """
    Get the Coordinates for a Tile in Lon-lat-grid.

    Args:
        tilename (str): 5 Letter MGRS tile

    Returns:
        tuple: Min-Lon, Min-Lat, Max-Lon, Max-Lat
    """    
    tile = COORDS[tilename]
    
    return tile["MinLon"], tile["MinLat"], tile["MaxLon"], tile["MaxLat"]

if __name__ == "__main__":
    main()