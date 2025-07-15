from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import os
import tkinter as TK
import random
import cv2
import math
import time
import re


def remove_lines(image):
    kernel = np.ones((1, int(image.shape[1]/10)), np.uint8)
    lines = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = np.array(cv2.add(image, (255 - lines)))
    kernel = np.ones((int(image.shape[0]/10), 1), np.uint8)
    lines = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = np.array(cv2.add(image, (255-lines)))
    return image


def remove_lines_vertical(image):
    kernel = np.ones((int(image.shape[0]/10), 1), np.uint8)
    lines = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = np.array(cv2.add(image, (255-lines)))
    return image


"""def brighten(image):
    kernel = np.ones((int(image.shape[0]/10), 1), np.uint8)
    lines = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = np.array(cv2.add(image, (255 - lines)))
    Image.fromarray(image.astype("uint8"), "L").show()

    kernel = np.ones((1, int(image.shape[1]/10)), np.uint8)
    lines = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    Image.fromarray(lines.astype("uint8"), "L").show()
    small_lines = Image.fromarray(
        lines.astype("uint8"), "L")
    small_lines.thumbnail((800, 600))
    ret, line = cv2.threshold(lines, 225, 255, 0)
    Image.fromarray(line.astype("uint8"), "L").show()
    lines = cv2.add(lines, (255 - line))
    Image.fromarray(lines.astype("uint8"), "L").show()
    image = np.array(cv2.add(image, (255 - lines)))
    Image.fromarray(image.astype("uint8"), "L").show()
    return image"""


def brighten(image):
    kernel = np.ones((50, 50), np.uint8)
    lines = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = np.array(cv2.add(image, (255 - lines)))
    return image


def search_element(r, c, image, labels, threshold, current_label):
    edge = [c, c + 1, r, r + 1]
    component = list()
    queue = [(r, c)]
    while len(queue) > 0:
        pos = queue[0]
        queue.remove(pos)
        if pos[0] >= 0 and pos[0] < image.shape[0] and pos[1] >= 0 and pos[1] < image.shape[1] and image[pos[0]][pos[1]] <= threshold and labels[pos[0], pos[1]] == 0:
            labels[pos[0]][pos[1]] = current_label
            component.append(pos)
            if pos[1] < edge[0]:
                edge[0] = pos[1]
            if pos[1]+1 > edge[1]:
                edge[1] = pos[1]+1
            if pos[0] < edge[2]:
                edge[2] = pos[0]
            if pos[0]+1 > edge[3]:
                edge[3] = pos[0]+1

            queue.append((pos[0], pos[1]+1))
            queue.append((pos[0]+1, pos[1]+1))
            queue.append((pos[0]+1, pos[1]))
            queue.append((pos[0]+1, pos[1]-1))
            queue.append((pos[0], pos[1]-1))
            queue.append((pos[0]-1, pos[1]-1))
            queue.append((pos[0]-1, pos[1]))
            queue.append((pos[0] - 1, pos[1] + 1))
    return edge, component


def get_components(image):
    labels = np.zeros(image.shape)
    current_label = 1
    intensity_threshold = get_threshold(image)
    edges = list()
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            if r >= 0 and r < image.shape[0] and c >= 0 and c < image.shape[1] and image[r][c] <= intensity_threshold and labels[r, c] == 0:
                edges.append(search_element(r, c, image, labels,
                                            intensity_threshold, current_label)[0])
                current_label += 1

    for component in edges:
        if component[3] - component[2] <= 1 or component[1] - component[0] <= 1 or (component[3] - component[2] <= 5 and component[1] - component[0] <= 5):
            edges.remove(component)
    return edges


def mean_height(boxes):
    sum = 0
    for box in boxes:
        sum += box[3] - box[2]
    return 0 if len(boxes) <= 0 else int(sum/len(boxes))


def mean_width(boxes):
    sum = 0
    for box in boxes:
        sum += box[1] - box[0]
    return 0 if len(boxes) <= 0 else int(sum/len(boxes))


def histogram(image):
    histo = [0] * 256
    for pixel in image:
        histo[pixel] += 1
    return histo


def get_threshold(image):
    pixelData = np.ravel(image).astype("int32")
    histData = histogram(pixelData)
    total = sum(pixelData.ravel())
    sumB = 0
    wB = 0
    wF = 0

    varMax = 0
    threshold = 0
    for t in range(255):
        wB += histData[t]
        if wB == 0:
            continue
        wF = len(pixelData) - wB
        if wF == 0:
            break

        sumB += t * histData[t]
        mB = sumB / wB
        mF = (total - sumB) / wF
        varBetween = wB * wF * (mB - mF) * (mB - mF)
        if varBetween > varMax:
            varMax = varBetween
            threshold = t

    return min(220, threshold)


def find_boxes(image, width, black):
    boxes = list()
    start = -1
    for j in range(0, image.shape[1], width):
        for i in range(image.shape[0]):
            if image[i][j] == 0 if black else image[i][j] == 255:
                if start == -1:
                    start = i
            elif start > -1:
                boxes.append((start, i))
                start = -1
        start = -1
    return boxes


def median_height(boxes, percentage):
    heights = list()
    for box in boxes:
        heights.append(box[1] - box[0])
    heights.sort()
    return 0 if len(heights) == 0 else heights[int(len(heights)*percentage)]


def paint(image, width, use_merge_height=False, merge_height=0, deletion_height=0, threshhold_bias=1.05):
    new_image = np.zeros(image.shape)
    for i in range(image.shape[0]):
        pixel_values = list()
        total = 0.0
        num = 0
        for j in range(image.shape[1]):
            if j != 0 and j % width == 0 or j == image.shape[1]-1:
                avg = int(float(total) / num)
                new_image[i][j-num:j+1] = avg
                pixel_values.clear()
                total = 0.0
                num = 0
            num += 1
            total += image[i][j]
    columns = np.split(new_image, range(width, image.shape[1], width), axis=1)

    Image.fromarray(new_image.astype("uint8"), "L").show()

    new_image = np.empty((image.shape[0], 0))
    for column in columns:
        # binarization
        cropped = np.delete(column, slice(0, 20), 0)
        cropped = np.delete(cropped, slice(
            cropped.shape[0] - 20, cropped.shape[0]), 0)
        threshold_inds = column < get_threshold(column) * threshhold_bias
        # (get_threshold(
        # column)*1.05 if not use_merge_height else get_threshold(cropped)*1.15)
        column = np.full(column.shape, 255)
        column[threshold_inds] = 0

        new_image = np.concatenate((new_image, column), axis=1)

    Image.fromarray(new_image.astype("uint8"), "L").show()

    median = median_height(find_boxes(new_image, width, False),
                           0.5)*0.6 if not use_merge_height else merge_height
    for j in range(0, new_image.shape[1], width):
        # fill small white spaces
        white_boxes = find_boxes(new_image[:, j:j + width], width, False)
        for box in white_boxes:
            if box[1] - box[0] < median:
                new_image[box[0]:box[1], j:j + width] = 0

    #Image.fromarray(new_image.astype("uint8"), "L").show()

    median = median_height(find_boxes(new_image, width, True),
                           0.65) if not use_merge_height else deletion_height
    for j in range(0, new_image.shape[1], width):
        # remove dangling and extra small black rectangles
        black_boxes = find_boxes(new_image[:, j:j + width], width, True)
        for box in black_boxes:
            if box[1] - box[0] < median/4 or (box[1] - box[0] < median and
                                              not (0 in new_image[box[0]:box[1], j] or 0 in new_image[box[0]:box[1], min(new_image.shape[1] - 1, j + width - 1)])):
                new_image[box[0]:box[1], j:j + width] = 255

    #Image.fromarray(new_image.astype("uint8"), "L").show()

    median = median_height(find_boxes(new_image, width, True),
                           0.65) if not use_merge_height else image.shape[0]/3
    for j in range(0, new_image.shape[1], width):
        # remove large black rectangles
        black_boxes = find_boxes(new_image[:, j:j + width], width, True)
        for box in black_boxes:
            if box[1] - box[0] > median * 2:
                new_image[box[0]:box[1], j:j + width] = 255
    return new_image


def dialate(image, width):
    kernel = np.ones((1, 4*width), np.uint8)
    return np.array(cv2.erode(image, kernel, iterations=1))


def remove_diagonals(line):
    slope = 1
    index = 0
    while abs(slope) >= 0.3 and index < len(line) - 5:
        slope = 1 if (line[index][1]-line[index+4][1]) == 0 else (line[index]
                                                                  [0] - line[index+4][0]) / (line[index][1] - line[index+4][1])
        index += 1

    if index < 30:
        line = line[index:]

    slope = 1
    index = len(line)-1
    while abs(slope) >= 0.3 and index > 5:
        slope = 1 if (line[index][1]-line[index-4][1]) == 0 else (line[index]
                                                                  [0] - line[index-4][0]) / (line[index][1] - line[index-4][1])
        index -= 1
    if len(line)-index < 30:
        line = line[:index]

    return line


def thin(image, line_spacing, disconnect=True):
    image = np.array(image, np.uint8)
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)

    ret, image = cv2.threshold(image, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv2.erode(image, element)
        temp = np.array(cv2.dilate(eroded, element), np.uint8)
        temp = np.array(cv2.subtract(image, temp), np.uint8)
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()

        zeros = size - cv2.countNonZero(image)
        if zeros == size:
            done = True

    if disconnect:
        skel = cv2.Sobel(skel, cv2.CV_64F, 0, 1, ksize=3)
    skel = 255 - skel
    Image.fromarray(skel.astype("uint8"), "L").show()
    _lines = np.full((skel.shape[0], skel.shape[1], 3), 255)
    lines = list()
    for c in range(skel.shape[1]):
        for r in range(skel.shape[0]):
            if skel[r][c] == 0:
                nodes = bfs(skel, (r, c))[0]
                start = nodes[len(nodes) - 1]
                nodes, parents = bfs(skel, start)
                for pixel in nodes:
                    skel[pixel[0], pixel[1]] = 255
                current = nodes[len(nodes) - 1]
                line = [current]
                while current != nodes[0]:
                    current = parents[nodes.index(current) - 1]
                    line.append(current)

                if len(line) < 5:
                    continue

                if line[0][1] > line[-1][1]:
                    line.reverse()

                last = 0
                p = 0
                while p < len(line):
                    if line[p][1] < line[last][1] and last != 0:
                        new_line = line[:last]
                        if len(new_line) > 5:
                            new_line = remove_diagonals(new_line)
                            if new_line[0][1] > new_line[-1][1]:
                                new_line.reverse()
                            lines.append(new_line)
                        line = line[last:]
                        line.reverse()
                        p = 0
                        last = 0
                        continue
                    last = p
                    p += 1

                line = remove_diagonals(line)

                if len(line) < 5:
                    continue

                if line[0][1] > line[-1][1]:
                    line.reverse()

                lines.append(line)

    for line in lines:
        color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255))
        for p in line:
            _lines[max(0, min(p[0], image.shape[0]-1)), max(
                0, min(p[1], image.shape[1] - 1))] = color
    #Image.fromarray(_lines.astype("uint8"), "RGB").show()
    _lines = np.full((skel.shape[0], skel.shape[1], 3), 255)

    lines.sort(key=lambda x: (x[0][1], -len(x)))

    i = 0
    while i < len(lines):
        l1 = lines[i]

        closestLine = lines[0]
        closestDist = math.inf
        while closestLine:
            closestLine = None
            closestDist = math.inf
            for l2 in lines:
                if l1 == l2:
                    continue
                dist = abs(l1[-1][0]-l2[0][0])*0.5 + \
                    abs(l1[-1][1]-l2[0][1])*0.5
                if l2[0][1] >= l1[-1][1]-10 and abs(l1[-1][0]-l2[0][0]) < line_spacing and abs(l1[-1][1]-l2[0][1]) < image.shape[1]/3 and dist < closestDist:
                    closestLine = l2
                    closestDist = dist
            if closestLine:
                l1.extend(connect(l1[-1], closestLine[0]))
                l1.extend(closestLine)
                lines.remove(closestLine)
                i = -1
        size = abs(l1[-1][1] - l1[0][1])
        if size < image.shape[1] / 3 and (l1[0][1] > image.shape[1] / 6 or size < image.shape[1] / 6):
            lines.remove(l1)
            i = -1
        elif size < image.shape[1]:
            lines[i] = [(l1[0][0], n) for n in range(0, l1[0][1])] + l1 + [(l1[-1][0], n)
                                                                           for n in range(l1[-1][1], image.shape[1])]
        i += 1

    lines.sort(key=lambda x: sum(p[0] for p in x)/len(x))
    lines_clone = list(lines)
    for l in range(len(lines_clone)):
        dup = True
        for p in lines_clone[l]:
            if l == 0 or p not in lines_clone[l - 1]:
                dup = False
                break
        if dup:
            lines.remove(lines_clone[l])

    for line in lines:
        color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255))
        for p in line:
            _lines[max(0, min(p[0], image.shape[0]-1)), max(
                0, min(p[1], image.shape[1] - 1))] = color
    #Image.fromarray(_lines.astype("uint8"), "RGB").show()

    # _lines = np.full((skel.shape[0], skel.shape[1]), 255)

    return _lines, lines


def connect(p1, p2):
    points = list()

    if abs(p2[1] - p1[1]) <= 1:
        for _y in range(p1[0], p2[0]):
            points.append((_y, p1[1]))
        return points

    if abs(p2[0] - p1[0]) <= 1:
        for x in range(p1[1], p2[1]):
            points.append((p1[0], x))
        return points

    deltax = p2[1] - p1[1]
    deltay = p2[0] - p1[0]
    deltaerr = abs(deltay / deltax)
    error = 0
    y = p1[0]
    for x in range(p1[1], p2[1]):
        points.append((y, x))
        error = error + deltaerr
        if error >= 0.5:
            y += 1 if p1[0] < p2[0] else -1
            error = error - 1.0
    return points


def distance(p1, p2):
    return abs(p1[1] - p2[1]), abs(p1[0] - p2[0])


def bfs(image, start):
    unvisited = [start]
    visited = list()
    parents = list()
    while len(unvisited) > 0:
        current = unvisited[0]
        visited.append(current)
        unvisited.remove(current)
        for x in range(-1, 2):
            for y in range(-1, 2):
                if (x == 0 and y == 0) or (abs(x) == abs(y)):
                    continue
                if not ((current[0] + y, current[1] + x) in visited) and not ((current[0] + y, current[1] + x) in unvisited) and current[1] + x >= 0 and current[1] + x < image.shape[1] and current[0] + y >= 0 and current[0] + y < image.shape[0] and image[current[0] + y][current[1] + x] < 100:
                    unvisited.append((current[0] + y, current[1] + x))
                    parents.append((current[0], current[1]))
    return visited, parents


def trace_contor(previous, start, image, element):
    im = np.full((image.shape[0], image.shape[1], 3), 255)
    for p in element:
        im[p[0], p[1]] = (0, 0, 0)

    im[start[0], start[1]] = (0, 255, 0)
    im[previous[0], previous[1]] = (255, 0, 0)

    contour = list()
    cycle = [(1, 0), (0, 1), (0, 1), (-1, 0),
             (-1, 0), (0, -1), (0, -1), (1, 0)]
    c = 0
    current = (start[0], start[1]-1)
    for i in range(0, 8):
        current = (current[0] + cycle[i][0], current[1] + cycle[i][1])
        if current == previous:
            c = i
            break

    stop = start
    stopC = 0

    c += 1
    if c >= len(cycle):
        c = 0
    current = (current[0] + cycle[c][0], current[1] + cycle[c][1])
    temp = list()
    while current != stop or c != stopC:
        is_contour = any([(current[0], current[1] - 1) in element,
                          (current[0]-1, current[1]) in element, (current[0]+1, current[1]) in element, (current[0], current[1]+1) in element])
        if current[0] >= 0 and current[0] < image.shape[0] and current[1] >= 0 and current[1] < image.shape[1] and image[current[0], current[1]] > 0 and any([(current[0] + n[0], current[1] + n[1]) in element for n in cycle]):
            temp.clear()
            if len(contour) == 1:
                stopC = c
                stop = current
            if current != start:
                contour.append(current)
                im[current[0], current[1]] = (150, 150, 150)
            current = (current[0] - cycle[c][0], current[1] - cycle[c][1])
            if c % 2 == 0:
                c -= 2
            else:
                c -= 1
            if c < 0:
                c = len(cycle) + c
            if len(contour) > 10000:
                Image.fromarray(im.astype("uint8"), "RGB").show()
                break
        else:
            temp.append(current)
            if len(temp) >= 8:
                print(len(element))
                contour.extend(temp)
                print("broke")
                break

            c += 1
            if c >= len(cycle):
                c = 0
        current = (current[0] + cycle[c][0], current[1] + cycle[c][1])
    return contour


def seperate_overlapping(image, lines, line_spacing):
    l = 0
    while l < len(lines):
        line = lines[l]
        if len(line) <= 0:
            lines.remove(line)
            l -= 1
            continue
        new_line = list()
        _count = -1
        end_point = line[-1]
        tracing = True
        for p in line:
            if not tracing or (len(new_line) > 0 and (p[1] < new_line[-1][1] or p == line[line.index(p) - 1])):
                if not tracing and p == end_point:
                    tracing = True
                continue
            new_line.append(p)
            if image[p[0]][p[1]] == 0:
                edge, element = search_element(int(p[0]), int(p[1]), image, np.zeros(
                    image.shape), 1, 1)
                if edge[3] - edge[2] > image.shape[0] / 4:
                    for pixel in element:
                        image[pixel[0], pixel[1]] = 255
                    continue
                p_ind = line.index(p)
                while (line[p_ind][0] - 1, line[p_ind][1] - 1) in element or (line[p_ind][0] - 1, line[p_ind][1]) in element or (line[p_ind][0] - 1, line[p_ind][1] + 1) in element or (line[p_ind][0], line[p_ind][1] - 1) in element or (line[p_ind][0], line[p_ind][1] + 1) in element or (line[p_ind][0] + 1, line[p_ind][1] - 1) in element or (line[p_ind][0] + 1, line[p_ind][1]) in element or (line[p_ind][0] + 1, line[p_ind][1] + 1) in element:
                    p_ind -= 1
                im = np.full((image.shape[0], image.shape[1], 3), 255)
                for v in element:
                    im[v[0], v[1]] = (0, 0, 0)
                contour = trace_contor(
                    line[p_ind], line[p_ind+1], image, element)
                # print(len(contour))
                if len(contour) == 0:
                    continue
                # if l == 0:
                    # for c in contour:
                    # print(c)
                count = 0
                last_index = line.index(p) + 1
                for c in contour:
                    if c in line:
                        if line.index(c) > last_index:
                            last_index = line.index(c)
                if last_index >= len(line):
                    end_point = p
                    print("uh oh")
                else:
                    end_point = line[last_index]
                # print (str(p) + " " + str(end_point))
                above = list()
                below = list()
                minYbelow = image.shape[0]
                maxYbelow = 0
                minYabove = image.shape[0]
                maxYabove = 0
                temp = list()
                tracing_above = False
                lastInLine = False
                for c in contour:
                    temp.append(c)
                    if c in line:
                        count += 1
                        if (count % 2 == 1 and not lastInLine) or c == end_point:
                            below.extend(temp)
                        else:
                            below.extend(connect(temp[0], c))
                        temp.clear()
                        lastInLine = True
                        if c == end_point:
                            break
                    elif lastInLine:
                        lastInLine = False
                contour.reverse()
                count = 0
                lastInLine = False
                for c in contour:
                    temp.append(c)
                    if c in line:
                        count += 1
                        if (count % 2 == 1 and not lastInLine) or c == end_point:
                            above.extend(temp)
                        else:
                            above.extend(connect(temp[0], c))
                        temp.clear()
                        lastInLine = True
                        if c == end_point:
                            break
                    elif lastInLine:
                        lastInLine = False
                for t in above:
                    if t[0] < minYabove:
                        minYabove = t[0]
                    if t[0] > maxYabove:
                        maxYabove = t[0]
                for t in below:
                    if t[0] < minYbelow:
                        minYbelow = t[0]
                    if t[0] > maxYbelow:
                        maxYbelow = t[0]
                if len(above) == 0:
                    maxYabove = 0
                    minYabove = 0
                elif len(below) == 0:
                    maxYbelow = 0
                    minYbelow = 0
                # print(str(maxYabove) + " " + str(minYabove) + " " + str(maxYbelow) + " " + str(minYbelow))
                # print(str(len(above)) + " " + str(len(below)))
                if (len(above) < 100 and maxYabove-minYabove < line_spacing * 0.75) or (len(below) < 100 and maxYbelow-minYbelow < line_spacing * 0.75):
                    # overlapping
                    # print("overlapping")
                    if maxYabove-minYabove < maxYbelow-minYbelow:
                        new_line.extend(above)
                    else:
                        new_line.extend(below)
                    # new_line.extend (contour)
                else:
                    min_width = line_spacing / 2
                    min_left = list()
                    min_right = list()
                    im = np.full((image.shape[0], image.shape[1], 3), 255)
                    for c in contour:
                        if c[0] < p[0] + line_spacing / 2 and c[0] > p[0] - line_spacing / 2 and c[1] < p[1] + line_spacing / 2 and c[1] > p[1] - line_spacing / 2:
                            width = 0
                            for i in range(c[1]+1, p[1] + line_spacing):
                                if (c[0], i) in element:
                                    width += 1
                                elif width > 0:
                                    break
                            if (c[0], c[1] + width + 1) in contour and width > 0 and width < min_width:
                                seperation_point = c
                                endsep_point = (c[0], c[1] + width+1)
                                revcontour = list(contour)
                                if contour.index(seperation_point) > len(contour) / 2:
                                    revcontour = contour.__reversed__()

                                if seperation_point == contour[0]:
                                    continue

                                left = list()
                                right = list()
                                left_max = [
                                    seperation_point[0], seperation_point[0], seperation_point[1], seperation_point[1]]
                                right_max = [
                                    endsep_point[0], endsep_point[0], endsep_point[1], endsep_point[1]]
                                seperated = False
                                tracing_left = True
                                for k in revcontour:
                                    if not seperated:
                                        if tracing_left:
                                            left.append(k)
                                            if k[0] < left_max[0]:
                                                left_max[0] = k[0]
                                            if k[0] > left_max[1]:
                                                left_max[1] = k[0]
                                            if k[1] < left_max[2]:
                                                left_max[2] = k[1]
                                            if k[1] > left_max[3]:
                                                left_max[3] = k[1]
                                            if k == seperation_point:
                                                seperated = True
                                                tracing_left = False
                                        else:
                                            right.append(k)
                                            if k[0] < right_max[0]:
                                                right_max[0] = k[0]
                                            if k[0] > right_max[1]:
                                                right_max[1] = k[0]
                                            if k[1] < right_max[2]:
                                                right_max[2] = k[1]
                                            if k[1] > right_max[3]:
                                                right_max[3] = k[1]
                                            if k == end_point:
                                                break
                                    elif k == endsep_point:
                                        seperated = False
                                        if k == end_point:
                                            break
                                if left_max[1] - left_max[0] < line_spacing and left_max[3] - left_max[2] < line_spacing and right_max[1] - right_max[0] < line_spacing and right_max[3] - right_max[2] < line_spacing:
                                    min_width = width
                                    min_left = left
                                    min_right = right

                    if len(min_left) > 0 and len(min_right) > 0:
                        for P in min_left:
                            im[P[0], P[1]] = (255, 0, 0)
                        for P in min_right:
                            im[P[0], P[1]] = (0, 255, 0)
                        if len(min_left) > 2:
                            new_line.extend(min_left)
                        new_line.extend([(min_left[-1][0], i) for i in range(
                            min_left[-1][1], int(min_left[-1][1] + min_width + 1))])
                        if len(min_right) > 3:
                            new_line.extend(min_right)
                        # Image.fromarray(im.astype("uint8"), "RGB").show()

                tracing = False

        lines[l] = new_line
        l += 1
    return lines


def partionLines(image, raw_im, lines, raw_scale=1):
    # Image.fromarray(image.astype("uint8"), "RGB").show()
    lines_imgs = list()
    lines_imgs_raw = list()
    boxes = list()
    lines.insert(0, [(0, n) for n in range(0, image.shape[1])])
    lines.append([(image.shape[0] - 1, n) for n in range(0, image.shape[1])])
    if len(lines) < 3:
        return
    for l in range(len(lines) - 1):
        if len(lines[l]) <= 0:
            continue

        minh = min(p[0] for p in lines[l])
        maxh = max(p[0] for p in lines[l + 1])
        height = maxh - minh
        if height < 1:
            continue
        line_img = np.full((height, image.shape[1]), 255)
        line_im_raw = np.full(
            (int(height*raw_scale), raw_im.shape[1]), 255)

        lines[l].sort(key=lambda p: (p[1], -p[0]))
        lines[l+1].sort(key=lambda p: (p[1], -p[0]))
        top_ind = 0
        bottom_ind = 0
        for x in range(image.shape[1]):
            while lines[l][top_ind][1] < x and top_ind < len(lines[l])-1:
                top_ind += 1
            top = max(0, lines[l][top_ind][0])
            while lines[l + 1][bottom_ind][1] < x and bottom_ind < len(lines[l + 1])-1:
                bottom_ind += 1
            bottom = min(lines[l + 1][bottom_ind][0], image.shape[0])
            if top < bottom:
                line_img[top-minh:bottom-minh, x] = image[top:bottom, x]
                lowesty_raw = top
                ind = top_ind
                while ind >= 0 and lines[l][ind][1] > lines[l][top_ind][1] - raw_scale / 2:
                    if lines[l][ind][0] > lowesty_raw:
                        lowesty_raw = lines[l][ind][0]
                    ind -= 1
                ind = top_ind
                while ind < len(lines[l]) and lines[l][ind][1] < lines[l][top_ind][1] + raw_scale / 2:
                    if lines[l][ind][0] > lowesty_raw:
                        lowesty_raw = lines[l][ind][0]
                    ind += 1
                for y in range(int(lowesty_raw * raw_scale), int(bottom * raw_scale)):
                    try:
                        line_im_raw[int(y-(minh*raw_scale)), int((x-raw_scale/2) * raw_scale):int((x + raw_scale/2) *
                                                                                                  raw_scale)] = raw_im[y, int((x-raw_scale/2)*raw_scale):int((x + raw_scale/2) *
                                                                                                                                                             raw_scale)]
                    except:
                        print(str(int(y-(minh*raw_scale))) +
                              " " + str(y))
                        raise

        cropped_height1y = 0
        for r in range(line_img.shape[0]):
            for c in range(line_img.shape[1]):
                if line_img[r, c].all() == 0:
                    edge = search_element(r, c, line_img, np.zeros(
                        (image.shape[0], image.shape[1])), 100, 1)[0]
                    if edge[1]-edge[0] > 1 and edge[3]-edge[2] > 1:
                        cropped_height1y = r
                        break
            if cropped_height1y > 0:
                break
        cropped_height1x = 0
        for c in range(line_img.shape[1]):
            for r in range(line_img.shape[0]):
                if line_img[r, c].all() == 0:
                    edge = search_element(r, c, line_img, np.zeros(
                        (image.shape[0], image.shape[1])), 100, 1)[0]
                    if edge[1]-edge[0] > 1 and edge[3]-edge[2] > 1:
                        cropped_height1x = c
                        break
            if cropped_height1x > 0:
                break
        cropped_height2y = 0
        for r in range(line_img.shape[0]-1, 0, -1):
            for c in range(1, line_img.shape[1]):
                if line_img[r, c].all() == 0:
                    edge = search_element(r, c, line_img, np.zeros(
                        (image.shape[0], image.shape[1])), 100, 1)[0]
                    if edge[1]-edge[0] > 1 and edge[3]-edge[2] > 1:
                        cropped_height2y = r
                        break
            if cropped_height2y > 0:
                break
        cropped_height2x = 0
        for c in range(line_img.shape[1]-1, 0, -1):
            for r in range(1, line_img.shape[0]):
                if line_img[r, c].all() == 0:
                    edge = search_element(r, c, line_img, np.zeros(
                        (image.shape[0], image.shape[1])), 100, 1)[0]
                    if edge[1]-edge[0] > 1 and edge[3]-edge[2] > 1:
                        cropped_height2x = c
                        break
            if cropped_height2x > 0:
                break
        if cropped_height2y - cropped_height1y > 1 and cropped_height2x - cropped_height1x > 1:
            line_im_raw = np.delete(line_im_raw, slice(
                round((cropped_height1y-1)*raw_scale)), 0)
            line_im_raw = np.delete(line_im_raw, slice(
                round((cropped_height2y - cropped_height1y + 3) * raw_scale), line_im_raw.shape[0]), 0)
            line_im_raw = np.delete(line_im_raw, slice(
                round((cropped_height1x-1)*raw_scale)), 1)
            line_im_raw = np.delete(line_im_raw, slice(
                round((cropped_height2x - cropped_height1x + 3) * raw_scale), line_im_raw.shape[1]), 1)
            lines_imgs_raw.append(Image.fromarray(
                line_im_raw.astype("uint8"), "L"))
            line_img = np.delete(line_img, slice(cropped_height1y-1), 0)
            line_img = np.delete(line_img, slice(
                cropped_height2y - cropped_height1y + 3, line_img.shape[0]), 0)
            line_img = np.delete(line_img, slice(cropped_height1x-1), 1)
            line_img = np.delete(line_img, slice(
                cropped_height2x - cropped_height1x + 3, line_img.shape[1]), 1)
            lines_imgs.append(Image.fromarray(
                line_img.astype("uint8"), "L"))
        boxes.append([cropped_height1x, minh+cropped_height1y,
                      cropped_height2x, minh+cropped_height2y])

    return lines_imgs, lines_imgs_raw, boxes


def segmentLines(image, rotation=0):
    image = image.convert("L")
    image.show()
    if image.size[0] < image.size[1]:
        rotation = 0
    raw = np.rot90(np.array(image), rotation)
    raw = brighten(raw)
    raw = remove_lines_vertical(raw)
    Image.fromarray(raw.astype("uint8"), "L").show()
    delta = 1
    if image.size[0] > image.size[1]:
        delta = image.size[0] / 800
    elif image.size[1] > image.size[0]:
        delta = image.size[1] / 600
    # print(image.size)
    # print(delta)
    image.thumbnail((800, 600))
    pixels = np.array(image)
    pixels = np.rot90(pixels, rotation)
    pixels = brighten(pixels)
    #Image.fromarray(pixels.astype("uint8"), "L").show()
    pixels = remove_lines(pixels)
    Image.fromarray(pixels.astype("uint8"), "L").show()
    width = mean_width(get_components(pixels))
    # print(len(get_components(pixels)))
    height = int(mean_height(get_components(pixels)) * 0.9)
    # print("w " + str(width))
    painted = paint(pixels, width)
    line_spacing = int(median_height(find_boxes(painted, width, False), 0.4))
    painted = paint(pixels, line_spacing)
    Image.fromarray(painted.astype("uint8"), "L").show()
    painted = dialate(painted, line_spacing)
    Image.fromarray(painted.astype("uint8"), "L").show()
    painted, lines = thin(painted, line_spacing)
    Image.fromarray(painted.astype("uint8"), "L").show()
    thresh = get_threshold(pixels)
    ret, pixels = cv2.threshold(pixels, thresh, 255, 0)
    pixels = np.pad(pixels, 1, "constant", constant_values=255)
    lines = seperate_overlapping(pixels, lines, line_spacing)
    painted = np.full((pixels.shape[0] - 2, pixels.shape[1] - 2, 3), 255)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    c = 0
    for line in lines:
        for p in line:
            painted[max(0, min(p[0]-1, painted.shape[0]-1)), max(
                0, min(p[1] - 1, painted.shape[1] - 1))] = colors[c]
        c += 1
        if c > 2:
            c = 0
    Image.fromarray(painted.astype("uint8"), "RGB").show()
    rgb_pixels = np.array(image.convert("RGB"))
    rgb_pixels = np.rot90(rgb_pixels, rotation)
    pasted = 255 - ((255 - rgb_pixels) + (255 - painted))
    pasted = np.clip(pasted, 0, 255)
    Image.fromarray(pasted.astype("uint8"), "RGB").show()

    pixels = pixels[1:-1, 1:-1]
    #pixels = cv2.cvtColor(pixels.astype("uint8"), cv2.COLOR_GRAY2RGB)
    #Image.fromarray(pixels.astype("uint8"), "L").show()
    # print(pixels.shape)
    lines_imgs, line_imgs_raw, coords = partionLines(pixels, raw, lines, delta)

    count = starting_count
    c = 0
    image = image.rotate(rotation*90, expand=True)
    for img in lines_imgs:
        count = segmentWords(img, line_imgs_raw[c], count, width,
                             height, image, coords[c])
        c += 1
    image.show()

    return line_imgs_raw


def segmentWords(image, raw, count, width, height, final, coords):
    image = image.convert("L")
    pixels = np.array(image)
    pixels = np.rot90(pixels, 3)
    raw = image.convert("L")
    raw = np.array(raw)
    raw = np.rot90(raw, 3)
    Image.fromarray(pixels.astype("uint8"), "L").show()
    print("w:" + str(width))
    print("h:" + str(height))
    if width == 0:
        return
    painted = paint(pixels, pixels.shape[1]-1, True,
                    int(height/(math.sqrt(width))*2.5),  int(height/(math.sqrt(width))*2.5)*3, 1.15)
    Image.fromarray(painted.astype("uint8"), "L").show()
    # print(int(mean_width(get_components(painted))))
    if 0 not in painted:
        return count
    painted, lines = thin(painted, height, False)
    pixels = np.pad(pixels, 1, "constant", constant_values=255)
    lines = seperate_overlapping(pixels, lines, height)
    painted = np.full((pixels.shape[0] - 2, pixels.shape[1] - 2, 3), 255)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    c = 0
    for line in lines:
        for p in line:
            painted[max(0, min(p[0]-1, painted.shape[0]-1)), max(
                0, min(p[1] - 1, painted.shape[1] - 1))] = colors[c]
        c += 1
        if c > 2:
            c = 0

    rgb_pixels = np.array(image.convert("RGB"))
    rgb_pixels = np.rot90(rgb_pixels, 3)
    pasted = 255 - ((255 - rgb_pixels) + (255 - painted))
    pasted = np.clip(pasted, 0, 255)
    Image.fromarray(pasted.astype("uint8"), "RGB").show()

    pixels = pixels[1:-1, 1:-1]
    words, words_raw, bbs = partionLines(pixels, raw, lines)
    c = 0
    for word in words:
        word = np.array(word)
        word = np.rot90(word)
        draw = ImageDraw.Draw(final)
        draw.rectangle([max(0, bbs[c][1] + coords[0] - 1), max(0, coords[3]-bbs[c][2] - 1),
                        min(final.size[0] - 1, coords[0] + bbs[c][3] + 1), min(final.size[1] - 1, coords[3] - bbs[c][0] + 1)], width=1)
        c += 1
        #word = Image.fromarray(word.astype("uint8"), "RGB")
        # word.save(os.path.join(os.path.dirname(__file__),
        # "handwriting words", "word" + str(count) + ".jpg"))
        count += 1
    return count


def sorted_alphanumeric(data):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

"""
listdir = sorted_alphanumeric(os.listdir(os.path.join(os.path.dirname(__file__),
                                                      "handwriting words")))
starting_count = int("".join(re.findall('\\d+', str(listdir[-1])))) + 1
"""

if __name__ == "__main__":
    
    test_image_path = "CustomHandwritingDataset/Forms/form4.jpg"
    test_image = Image.open(test_image_path).convert("L")
    segmentLines(test_image, -1)
