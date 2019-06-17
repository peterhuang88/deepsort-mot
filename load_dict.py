"""
load a dict of {id : [coords] }, do stuff with it
* visualize
* analyze
"""

import json
import cv2


# --- load dict ---
f = open("coords.json", 'r')
coord_dict = json.load(f)
f.close()

# with open(filename, 'r') as f
#  dict = json.load(f)

print("dict size: {}".format(len(coord_dict)))

# --- /load dict ---


# --- visualize ---

# open map
map2d = cv2.imread('../video/purdue/msee.png')

# try 1 id
pos_list = coord_dict['4']
print("id 1, list size: {}".format(len(pos_list)))


COLOR_BLUE = (0, 0, 255)
for pos in pos_list:
    print(pos)
    cv2.circle(map2d, (int(pos[0]), int(pos[1])), 1, COLOR_BLUE, -1)

while True:
    cv2.imshow('map', map2d)
    cv2.waitKey(1)















