import numpy as np
from PIL import Image, ImageDraw
import math

# Load data from a .npy file
data = np.load("./CustomHandwritingDataset/Strokes/Form1/form1-line4-strokes.npy")

minx = math.inf
miny = math.inf
maxx = -math.inf
maxy = -math.inf

points = []
stroke_indices = []  # indices where eos == 1, marking stroke ends

previous = (0, 0)
for i, row in enumerate(data):
    x = int(float(row[0]))
    y = int(float(row[1]))
    eos = int(float(row[2]))

    if x < minx:
        minx = x
    if y < miny:
        miny = y
    if x > maxx:
        maxx = x
    if y > maxy:
        maxy = y

    points.append((x, y))
    if eos == 1:
        stroke_indices.append(i)

width = maxx - minx + 1
height = maxy - miny + 1

# Define scale factor for higher resolution output
scale = 4

# Create white background image with scaled size
image = Image.new("RGB", (width * scale, height * scale), (255, 255, 255))
draw = ImageDraw.Draw(image)

radius = 6

# Function to interpolate color between start and end colors
def interpolate_color(start_color, end_color, t):
    return tuple(int(start_color[i] + (end_color[i] - start_color[i]) * t) for i in range(3))

start_idx = 0
for end_idx in stroke_indices:
    stroke_points = points[start_idx:end_idx+1]
    n_points = len(stroke_points)
    if n_points == 0:
        start_idx = end_idx + 1
        continue

    # Draw lines with color gradient from blue to red
    start_color = (0, 0, 255)  # Blue
    end_color = (255, 0, 0)    # Red

    for i in range(n_points - 1):
        t = i / (n_points - 1)
        line_color = interpolate_color(start_color, end_color, t)
        p1 = ((stroke_points[i][0] - minx) * scale, (height - 1 - (stroke_points[i][1] - miny)) * scale)
        p2 = ((stroke_points[i+1][0] - minx) * scale, (height - 1 - (stroke_points[i+1][1] - miny)) * scale)
        draw.line([p1, p2], fill=line_color, width=2 * scale)

    # Draw circles for points
    for i, (x, y) in enumerate(stroke_points):
        cx = (x - minx) * scale
        cy = (height - 1 - (y - miny)) * scale
        leftUpPoint = (cx - radius, cy - radius)
        rightDownPoint = (cx + radius, cy + radius)
        if i == 0:
            # Start point in green
            draw.ellipse([leftUpPoint, rightDownPoint], fill=(0, 255, 0))
        elif i == n_points - 1:
            # End point in red
            draw.ellipse([leftUpPoint, rightDownPoint], fill=(255, 0, 0))
        #else:
            # Intermediate points in black
            #draw.ellipse([leftUpPoint, rightDownPoint], fill=(0, 0, 0))

    start_idx = end_idx + 1

print(f"[{minx},{maxx}] [{miny},{maxy}]")
image.show()
