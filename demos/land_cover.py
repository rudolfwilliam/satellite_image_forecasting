import tifffile
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from os.path import join
import utm
import earthnet as en
import statistics as st
#import pymgrs
#import geopandas
import geopandas as gpd
import rasterio

from pyproj import Proj, transform, Transformer

ESA_num_pixes = 36000

def convert_mgrs_to_gps(mgrs_coordinates):
    hemisphere, rest = mgrs_coordinates[0], mgrs_coordinates[1:]
    
    # Check if the hemisphere is 'N' for northern or 'S' for southern
    is_northern = hemisphere.upper() == 'N'

    # Split the rest of the coordinates into zone, easting, and northing
    zone, rest = rest[:2], rest[2:]
    easting, northing = float(rest[len(rest)//2:]), float(rest[:len(rest)//2])

    # Adjust easting based on whether it's east or west
    if mgrs_coordinates.startswith('W'):
        easting = -easting

    # Convert UTM to latitude and longitude
    latitude, longitude = utm.to_latlon(easting, northing, int(zone), is_northern)

    return latitude, longitude

'''# Example usage:
mgrs_coordinates_example = "N30W120"
latitude, longitude = convert_mgrs_to_gps(mgrs_coordinates_example)
print("Latitude:", latitude)
print("Longitude:", longitude)

wgs84 = Proj(init='epsg:4326')'''

datacube = '''C:/Users/Oto/Documents/GitHub/drought_impact_forecasting/Data/train/29SND/29SND_2020-01-01_2020-05-29_3385_3513_4921_5049_52_132_76_156.npz'''
ex_dc = '''C:/Users/Oto/Documents/GitHub/drought_impact_forecasting/Data/extreme/extreme_test_split/context/32UQC/context_32UQC_2018-01-28_2018-11-23_5305_5433_3257_3385_82_162_50_130.npz'''

ex_dir='''C:/Users/Oto/Documents/GitHub/drought_impact_forecasting/Data/extreme/extreme_test_split/context/32UQC'''
ex_dcs = os.listdir(ex_dir)
ex_dcs = [dc[8:] for dc in ex_dcs]
print(ex_dcs)

'''
for dc in ex_dcs:
    print(en.get_coords_from_cube(dc, return_meso=True))
    
    sample = en.cube_gallery(join(ex_dir, 'context_' + dc), 'rgb')
    plt.show()
'''






lc_files_dir='C:/Users/Oto/Downloads/ESA_WorldCover_10m_2021_v200_60deg_macrotile_N30E000'
lc_files = os.listdir(lc_files_dir)
lc_file_tiles = []



##############################

# Test geopd
#print(lc_files[0])
#tif_file = rasterio.open(join(lc_files_dir, lc_files[0]))
first_dc = 'ESA_WorldCover_10m_2021_V200_N51E012_Map.tif'
tif_file = rasterio.open(join(lc_files_dir, first_dc))
data = tif_file.read()

def get_ESA_file(N, E):
    nor = int(N // 3 * 3)
    eas = int(E // 3 * 3)
    return "N" + str(nor).zfill(2) + "E" + str(eas).zfill(3)

def get_rel_ESA_coord(N, E):
    rel_N = round(((N % 3) / 3) * ESA_num_pixes)
    rel_E = round(((E % 3) / 3) * ESA_num_pixes)
    return rel_N, rel_E

def get_lc_for_pixel(rel_N, rel_E, data):
    return data[0, ESA_num_pixes-1-rel_N, rel_E]



# Get relative position in 3x3 degree area
# lower left of EN dc
low_left_N = 51.65910012808779
low_left_E = 13.363196663627646

ESA_file = get_ESA_file(low_left_N, low_left_E)

rel_ESA_N, rel_ESA_E  = get_rel_ESA_coord(low_left_N, low_left_E)

pixel_lc = get_lc_for_pixel(rel_ESA_N, rel_ESA_E, data)

# Getting land cover for whole EN datacube
cube_gps = en.get_coords_from_cube(ex_dcs[0], return_meso=False)
low_left_N = cube_gps[1]
low_left_E = cube_gps[0]
up_right_N = cube_gps[3]
up_right_E = cube_gps[2]

whole_matrix = np.zeros((128, 128, 3))
for i in range(128):

np.linspace(low_left_N, up_right_N, 128)


testN, testE = get_rel_ESA_coord(51.3524486, 12.3967362)


for i in range(testN-10, testN+10):
    print(data[0, i, testE-10:testE+10])

    #for j in range(testE-10, testE+10):


for i in range(36000):
    for j in range(36000):
        if data[0,i,j] != 60:
            print(i)
            print(j)
            break



df_tile = gpd.read_file(join(lc_files_dir, lc_files[0]))


###############################






# We assume they are all in the north
# Get lower left
# lat is north / south, long is west / east
def get_lat_long(tiles_str):
    lat = int(tiles_str[1:3])
    long = int(tiles_str[4:7])
    return lat, long

# Find location of EN datacube
file = ex_dc.split('/')[-1][:-4]

if 'context' in file:
    file = file[8:]

print(file)
tile = file.split('_')[0]
coord_components = file.split('_')[-8:]
print(coord_components)

print(en.get_coords_from_cube(file, return_meso=True))
# For 1st EN cube ('32UQC_2018-01-28_2018-11-23_5305_5433_3257_3385_82_162_50_130'), we need the ESA tile N51E12



# Returns:
# (13.363196663627646, 51.65910012808779, 13.402326100129972, 51.68067103247151, 11.892704219008172, 52.290067824724105, 11.9170767884844, 52.30385735283948)
# Based on docu: hrxmin, hrxmax, hrymin, hrymax, mesoxmin, mesoxmax, mesoymin, mesoymax
# but we have:
#   51.65910012808779
#   51.68067103247151
#   52.290067824724105
#   52.30385735283948

# And then
#   11.892704219008172
#   11.9170767884844
#   13.363196663627646
#   13.402326100129972

print(en.get_coords_from_cube(file, return_meso=False))
# Based on docu: hrxmin, hrxmax, hrymin, hrymax
# (13.363196663627646, 51.65910012808779, 13.402326100129972, 51.68067103247151)
#   51.65910012808779
#   51.68067103247151

#   13.363196663627646
#   13.402326100129972

# The first 2 of each give: (South Brandenburg, Germany). This is the bottom left
# https://earth.google.com/web/search/51.65910012808779,+13.363196663627646/@51.66906257,13.36901207,102.36291195a,7851.78670397d,35y,0h,0t,0r/data=CmkaPxI5GSklX2Rd1ElAIb4H-er0uSpAKiU1MS42NTkxMDAxMjgwODc3OSwgMTMuMzYzMTk2NjYzNjI3NjQ2GAIgASImCiQJY4zvdTavNEARZIzvdTavNMAZVJ51TzKbSUAhUJ51TzKbScA6AwoBMA

# The second two of each give: (South Brandenburg, Germany). This is the top right
# https://earth.google.com/web/search/51.68067103247151,+13.402326100129972/@51.67518565,13.39132361,106.78059971a,8201.28413519d,35y,0h,0t,0r/data=CmkaPxI5GfUqMjog10lAIYqnw6_9zSpAKiU1MS42ODA2NzEwMzI0NzE1MSwgMTMuNDAyMzI2MTAwMTI5OTcyGAIgASImCiQJkmH05bldS0ARSOd_ifBySEAZ3fomS2wTOEAh9kJCIflDB0A6AwoBMA

# To visialize the datacube:
trainSample = en.cube_gallery(ex_dc, 'rgb')
plt.show()

# For the mesoscale coords we have: (seems a bit messed up)
# 52.290067824724105, 11.892704219008172 gives us the top left corner (Sachsen-Anhalt, just North East of Magedeburg)
# 52.30385735283948, 11.9170767884844  gives again something in the top left (Sachsen-Anhalt, just North East of Magedeburg), but north east of the previous point




for f in lc_files:
    tiles = []
    tiles_str = f.split('_')[-2]
    lat, lon = get_lat_long(tiles_str)

    utm_zone = int((lon + 180) / 6) + 1
    wgs84 = Proj(init='epsg:4326')
    easting, northing = transform(wgs84, Proj(proj='utm', zone=utm_zone, ellps='WGS84'), lon, lat)
    #utm = Proj(proj='utm', zone=utm_zone, ellps='WGS84')

    transformer = Transformer.from_proj(Proj(proj='utm', zone=utm_zone, ellps='WGS84'), Proj(proj='latlong', datum='WGS84'))
    mgrs_coordinates = transformer.transform(easting, northing)


    #mgrs_coordinates = transform(utm, Proj(proj='utm', zone=utm_zone, ellps='WGS84', preserve_units=False), easting, northing, inverse=True)



    print(mgrs_coordinates)
    print("Done")

    


# Get corresponding lc_file
cube_tile = datacube.split('/')[-1].splt('_')
print('Cube is in tile: ' + cube_tile)






