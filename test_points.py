import numpy as np
from PIL import Image
import re
import math

f = open("prime.txt", "r")

image = np.full((200, 1000), 255)
minx = math.inf
miny = math.inf
maxx = 0
maxy = 0

previous = (100, 100)
for l in f:
    line = l.split()
    x = int(float(line[0])) + previous[0]
    y = int(float(line[1])) + previous[1]
    eos = int(float(line[2]))

    if x < minx:
        minx = x
    if y < miny:
        miny = y
    if x > maxx:
        maxx = x
    if y > maxy:
        maxy = y

    previous = (x, y)

    image[-y, x] = 0

print("[" + str(minx) + "," + str(maxx) + "]" +
      " " + "[" + str(miny) + "," + str(maxy) + "]")
Image.fromarray(image.astype("uint8"), "L").show()
