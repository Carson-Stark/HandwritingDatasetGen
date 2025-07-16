import cv2
import numpy as np
import math
import random
from PIL import Image, ImageDraw
import time
import tkinter as tk


def trace_contor(previous, start, image, element):
    im = np.full((image.shape[0], image.shape[1], 3), 255)
    for p in element:
        im[p[0], p[1]] = (0, 0, 0)

    im[start[0], start[1]] = (0, 255, 0)
    im[previous[0], previous[1]] = (255, 0, 0)

    contour = [start]
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
        is_contour = (current[0], current[1]) in element
        if image[current[0], current[1]] < 100 and any([image[current[0] + n[0], current[1] + n[1]] > 200 for n in cycle]):
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
                #Image.fromarray(im.astype("uint8"), "RGB").show()
                break
        else:
            temp.append(current)
            if len(temp) >= 8:
                print(len(element))
                contour.extend(temp)
                #Image.fromarray(im.astype("uint8"), "RGB").show()
                print("broke")
                break

            c += 1
            if c >= len(cycle):
                c = 0
        current = (current[0] + cycle[c][0], current[1] + cycle[c][1])
    return contour


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


def check_kernels(image, pixel, kernels, require_all=False):
    hits = 0
    for k in kernels:
        hit = True
        for x in range(pixel[1] - int(k.shape[1] / 2), pixel[1] + int(k.shape[1] / 2)):
            for y in range(pixel[0] - int(k.shape[0] / 2), pixel[0] + int(k.shape[0] / 2)):
                if image[y, x] < 100 and k[y - (pixel[0] - int(k.shape[0] / 2)), x - (pixel[1] - int(k.shape[1] / 2))] == 1:
                    hit = False
                if image[y, x] >= 100 and k[y - (pixel[0] - int(k.shape[0] / 2)), x - (pixel[1] - int(k.shape[1] / 2))] == -1:
                    hit = False
        if hit and not require_all:
            # print(k)
            return True
        elif hit:
            hits += 1
    if hits == len(kernels):
        print("all")
        return True
    return False


def convert4connected(image):
    kernels = [np.array((
        [0, 0, 0],
        [-1, 1, 0],
        [1, -1, 0]), dtype="int")]
    cop_kern = list(kernels)
    for k in cop_kern:
        rotations = list()
        for r in range(1, 4):
            rotations.append(np.rot90(k, r))
        kernels.extend(rotations)

    image = np.array(image, np.uint8)
    additions = np.zeros(image.shape, np.uint8)
    # Image.fromarray(image.astype("uint8"), "L").show()

    for k in kernels:
        additions += cv2.morphologyEx(image,
                                      cv2.MORPH_HITMISS, k).astype(np.uint8)
    additions[0, :] = 0
    additions[-1, :] = 0
    additions[:, 0] = 0
    additions[:, -1] = 0
    image -= additions

    # Image.fromarray(image.astype("uint8"), "L").show()

    return image


def remove_holes(image):
    kernel = np.array((
        [-1, -1, -1],
        [-1, 1, -1],
        [-1, -1, -1]), dtype="int")
    kernels = [np.array((
        [-1, -1, -1],
        [-1, 1, 1],
        [-1, -1, -1]), dtype="int")]
    cop_kern = list(kernels)
    for k in cop_kern:
        rotations = list()
        for r in range(1, 4):
            rotations.append(np.rot90(k, r))
        kernels.extend(rotations)
    kernels.append(kernel)
    im_cop = image.copy()
    for k in kernels:
        image -= cv2.morphologyEx(im_cop,
                                  cv2.MORPH_HITMISS, k).astype(np.uint8)
    #Image.fromarray(image.astype("uint8"), "L").show()

    return image


def seperate_letters(image):
    #Image.fromarray(image.astype("uint8"), "L").show()

    comp_widths = list()
    elements = list()
    labels = np.zeros(image.shape)
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            if image[r, c] == 0 and labels[r, c] == 0:
                edge, element = search_element(r, c, image, labels, 100, 1)
                if len(element) > 10:
                    comp_widths.append(edge[1] - edge[0])
                elements.append((edge, element))
    comp_widths.sort()
    median_width = comp_widths[round(len(comp_widths) / 2)]

    # print(comp_widths)
    # print("median: " + str(median_width))
    dis_im = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for e in elements:
        width = e[0][1] - e[0][0]

        if width > median_width * 3:
            overlap_average = 0
            average_height = 0
            for x in range(e[0][0], e[0][1]+1):
                previous_overlap = False
                height = 0
                for y in range(e[0][2], e[0][3]+1):
                    if (y, x) in e[1]:
                        height += 1
                        if not previous_overlap:
                            overlap_average += 1
                            previous_overlap = True
                    else:
                        previous_overlap = False
                average_height += height
            average_height /= e[0][1] - e[0][0]
            overlap_average /= e[0][1] - e[0][0]
            # print(overlap_average)
            if round(overlap_average) == 1 and average_height < median_width / 4:
                # print("underline")
                for p in e[1]:
                    image[p[0], p[1]] = 255
            continue
        elif (width > median_width/2 and e[0][3] - e[0][2] < median_width / 4) or width < 5 or e[0][3] - e[0][2] < 5:
            for p in e[1]:
                image[p[0], p[1]] = 255
            continue

        if width > median_width * 1.75:
            divisions = round(width / median_width)
            scores = list()
            connection_groups = list()
            for d in range(divisions-1, divisions+1):
                if d <= 1:
                    continue
                # print("div: " + str(d))
                div_size = math.ceil(width / d)
                # print("size: " + str(div_size))
                score = 0
                connections = list()
                for x in range(e[0][0] + div_size, e[0][1], div_size):
                    dis_im[:, x] = (255, 0, 0)
                    prev = (0, x)
                    start = (1, x)
                    for y in range(1, e[0][3]+1):
                        if image[y, x] == 0:
                            start = (y, x)
                            prev = (y - 1, x)
                            break
                    # find intersect groups
                    contour = trace_contor(prev, start, image, e[1])
                    intersect_groups = list()
                    group = list()
                    has_x = False
                    for p in contour:
                        if abs(p[1] - x) < div_size/2:
                            group.append(p)
                            if contour.index(p) == len(contour) - 1 and len(intersect_groups) > 0:
                                intersect_groups[0] = group + \
                                    intersect_groups[0]
                            if p[1] == x:
                                has_x = True
                        elif len(group) > 0:
                            if has_x:
                                intersect_groups.append(group)
                            has_x = False
                            group = list()
                        smooth = get_edge_smoothness(
                            contour, contour.index(p), 3)
                    """for g in intersect_groups:
                        color = (random.randint(0, 255), random.randint(
                            0, 255), random.randint(0, 255))
                        for p in g:
                            dis_im[p[0], p[1]] = color"""
                    # print("i " + str(len(intersect_groups)))

                    # make connections
                    cop_groups = list(intersect_groups)
                    con_scores = list()
                    potential_cons = list()
                    for g1 in cop_groups:
                        if g1 not in intersect_groups:
                            continue

                        # find group pair
                        pair = None
                        point_pair = None
                        min_dist = math.inf
                        for g2 in intersect_groups:
                            if g1 == g2:
                                continue
                            pp = (g1[0], g2[0])
                            dist = math.inf
                            # print(str(cop_groups.index(g1)) +
                            #      " " + str(cop_groups.index(g2)))
                            for p1 in g1:
                                for p2 in g2:
                                    smooth1 = get_edge_smoothness(
                                        contour, contour.index(p1), 5)
                                    if p1 == p2:
                                        smooth2 = get_edge_smoothness(
                                            list(reversed(contour)), list(reversed(contour)).index(p2), 3)
                                    else:
                                        smooth2 = get_edge_smoothness(
                                            contour, contour.index(p2), 3)
                                    # print(str(smooth1) + " " + str(smooth2))
                                    """if smooth1 > 0.4:
                                        dis_im[p1[0], p1[1]] = (255, 0, 0)
                                    if smooth2 > 0.4:
                                        dis_im[p2[0], p2[1]] = (255, 0, 0)"""
                                    # d = math.sqrt(math.pow(
                                    # p1[0] - p2[0], 2) + math.pow((p1[1] - p2[1]), 2)) -
                                    d = distance(p1, p2) if abs(p1[1] - p2[1]) < abs(p1[0] - p2[0]) else math.sqrt(
                                        math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2)*5)
                                    d -= min(smooth1, smooth2) * 5
                                    g1_x_ind = None
                                    for i in range(len(g1)):
                                        if g1[i][1] == x and (g1_x_ind == None or g1[g1_x_ind][0] < g1[i][0]):
                                            g1_x_ind = i
                                    g2_x_ind = None
                                    for i in range(len(g2)):
                                        if g2[i][1] == x and (g2_x_ind == None or g2[g2_x_ind][0] < g2[i][0]):
                                            g2_x_ind = i

                                    d += min(abs(g1.index(p1)-g1_x_ind)/len(g1),
                                             abs(g2.index(p2) - g2_x_ind) / len(g2)) * 5

                                    # d += abs((p1[1]+p2[1])/2 - x) / 5
                                    if d < dist and (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)) in e[1]:
                                        pp = (p1, p2)
                                        dist = d
                                        # print(d)
                                        """dis_im[p1[0], p1[1]] = (0, 255, 0)
                                        dis_im[p2[0], p2[1]] = (0, 255, 0)
                                        Image.fromarray(dis_im.astype(
                                            "uint8"), "RGB").show()
                                        dis_im[p1[0], p1[1]] = (0, 0, 0)
                                        dis_im[p2[0], p2[1]] = (0, 0, 0)"""
                            # print("d " + str(dist))
                            if dist < min_dist:
                                pair = g2
                                point_pair = pp
                                min_dist = dist
                        if pair == None:
                            continue

                        if len(g1) > 4 and len(pair) > 4:
                            if distance(pair[-1], g1[0]) < distance(pair[0], g1[0]):
                                pair = list(reversed(pair))
                            translation = g1[0][0] - pair[0][0]
                            index_delta = len(pair) / len(g1)
                            total_distance = 0
                            for i in range(2, len(g1)-2):
                                total_distance += abs(g1[i][0] - (
                                    pair[int(i * index_delta)][0] + translation))
                            total_distance /= len(g1) - 4
                        else:
                            total_distance = 0

                        for p in g1:
                            dis_im[p[0], p[1]] = (
                                (g1.index(p)/len(g1))*255, 0, 0)
                        for p in pair:
                            dis_im[p[0], p[1]] = (
                                0, 0, (pair.index(p)/len(pair))*255)

                        #Image.fromarray(dis_im.astype("uint8"), "RGB").show()

                        # check if valid
                        start_ind = contour.index(point_pair[0])
                        end_ind = len(contour) - \
                            (list(reversed(contour)).index(point_pair[1])) - 1
                        x1 = [math.inf, 0]
                        y1 = [math.inf, 0]
                        i = start_ind
                        while i != end_ind:
                            # dis_im[contour[i][0], contour[i][1]] = (255, 0, 0)
                            if contour[i][1] < x1[0]:
                                x1[0] = contour[i][1]
                            if contour[i][1] > x1[1]:
                                x1[1] = contour[i][1]
                            if contour[i][0] < y1[0]:
                                y1[0] = contour[i][0]
                            if contour[i][0] > y1[1]:
                                y1[1] = contour[i][0]
                            i += 1
                            i %= len(contour)
                        x2 = [math.inf, 0]
                        y2 = [math.inf, 0]
                        i = start_ind
                        while i != end_ind:
                            # dis_im[contour[i][0], contour[i][1]] = (0, 0, 255)
                            if contour[i][1] < x2[0]:
                                x2[0] = contour[i][1]
                            if contour[i][1] > x2[1]:
                                x2[1] = contour[i][1]
                            if contour[i][0] < y2[0]:
                                y2[0] = contour[i][0]
                            if contour[i][0] > y2[1]:
                                y2[1] = contour[i][0]
                            i -= 1
                            if i < 0:
                                i = len(contour) + i
                        #print("total " + str(total_distance))

                        # seperate
                        if x1[1] - x1[0] > median_width * 0.8 and x2[1] - x2[0] > median_width * 0.8 \
                                and y1[1] - y1[0] > median_width * 0.5 and y2[1] - y2[0] > median_width * 0.5 \
                                and (total_distance > 5 or distance(point_pair[0], point_pair[1]) <= 2 or div_size > median_width):
                            con = connect(point_pair[0], point_pair[1])
                            potential_cons.append(con)
                            con_scores.append(min_dist)
                        else:
                            # print("invalid")
                            potential_cons.append(None)
                            con_scores.append(median_width)

                        dis_im[point_pair[0][0],
                               point_pair[0][1]] = (0, 255, 0)
                        dis_im[point_pair[1][0],
                               point_pair[1][1]] = (0, 255, 0)
                        #Image.fromarray(dis_im.astype("uint8"), "RGB").show()
                        dis_im = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    if len(con_scores) < 1:
                        continue
                    min_score = con_scores[0]
                    connection = potential_cons[0]
                    for i in range(len(con_scores)):
                        if con_scores[i] < min_score and potential_cons[i] != None:
                            min_score = con_scores[i]
                            connection = potential_cons[i]
                    score += min_score
                    dis_im = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    if connection != None:
                        connections.append(connection)
                        for p in connection:
                            dis_im[p[0], p[1]] = (0, 255, 0)
                    #Image.fromarray(dis_im.astype("uint8"), "RGB").show()
                scores.append(score / (d-1))
                connection_groups.append(connections)
            min_score = scores[0]
            connections = connection_groups[0]
            for i in range(len(scores)):
                if scores[i] < min_score and len(connection_groups[i]) > 0:
                    min_score = scores[i]
                    connections = connection_groups[i]
            # print(len(connections))
            for c in connections:
                for p in c:
                    image[p[0], p[1]] = 255
    #Image.fromarray(image.astype("uint8"), "L").show()
    return image


def thin(image, line_height):
    # print("thinning")

    image = 255 - np.array(image, np.uint8)
    #Image.fromarray(image.astype("uint8"), "L").show()
    #image = cv2.medianBlur(image, 3)
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)

    ret, image = cv2.threshold(image, get_threshold(image), 255, 0)
    eroded = image.copy()
    done = False

    line_im = cv2.cvtColor(255-image, cv2.COLOR_GRAY2RGB)
    line_im[line_height, :] = (0, 0, 255)
    #Image.fromarray(line_im.astype("uint8"), "RGB").show()

    """elements = list()
    labels = np.zeros(image.shape)
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            if image[r, c] == 0 and labels[r, c] == 0:
                edge, element = search_element(r, c, image, labels, 100, 1)
                elements.append((edge, element))
    for e in elements:
        if e[0][1] - e[0][0] < :"""

    skel = cv2.ximgproc.thinning(image)

    connectivity_kernels = [np.array((
        [0, 1, 0],
        [0, 1, 1],
        [-1, 0, 0]), dtype="int")]
    cop_kern = list(connectivity_kernels)
    for k in cop_kern:
        rotations = list()
        for r in range(1, 4):
            rotations.append(np.rot90(k, r))
        connectivity_kernels.extend(rotations)
    for k in connectivity_kernels:
        c = cv2.morphologyEx(skel,
                             cv2.MORPH_HITMISS, k)
        skel = np.array(255 - (255 - skel + c), np.uint8)

    #Image.fromarray((255-skel).astype("uint8"), "L").show()

    """sk = np.array(skel)
    for r in range(1, skel.shape[0]-1):
        for c in range(1, skel.shape[1]-1):
            if check_kernels(skel, (r, c), connectivity_kernels, False):
                skel[r, c] = 0"""

    junction_kernels = [np.array((
        [0, 1, 0],
        [-1, 1, -1],
        [1, -1, 1]), dtype="int"),
        np.array((
            [1, -1, 0],
            [-1, 1, -1],
            [1, -1, 1]), dtype="int"),
        np.array((
            [0, -1, 1],
            [1, 1, -1],
            [0, 1, 0]), dtype="int")]
    cop_kern = list(junction_kernels)
    for k in cop_kern:
        rotations = list()
        for r in range(1, 4):
            rotations.append(np.rot90(k, r))
        junction_kernels.extend(rotations)
    skel_cop = np.array(skel)
    j = np.zeros(skel.shape)
    for k in junction_kernels:
        j += cv2.morphologyEx(skel_cop,
                              cv2.MORPH_HITMISS, k)

    ep_kernels = [np.array((
        [0, 1, 0],
        [-1, 1, -1],
        [-1, -1, -1]), dtype="int"),
        np.array((
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, -1]), dtype="int")]
    cop_kern = list(ep_kernels)
    for k in cop_kern:
        rotations = list()
        for r in range(1, 4):
            rotations.append(np.rot90(k, r))
        ep_kernels.extend(rotations)
    skel_cop = np.array(skel)
    e = np.zeros(skel.shape)
    for k in ep_kernels:
        e += cv2.morphologyEx(skel_cop,
                              cv2.MORPH_HITMISS, k)

    skel = 255 - skel
    # skel = cv2.cvtColor(skel, cv2.COLOR_RGB2GRAY)
    # skel = cv2.threshold(skel, 200, 255, cv2.THRESH_BINARY)[1]
    """skel = cv2.cvtColor(skel, cv2.COLOR_GRAY2RGB)
    skel[:, :, 0] += e.astype("uint8")
    skel[:, :, 1] += j.astype("uint8")
    endpoints = list()
    for r in range(skel.shape[0]):
        for c in range(skel.shape[1]):
            if (skel[r, c] == (255, 0, 0)).all():
                endpoints.append((r, c))"""

    skel_img = cv2.cvtColor(skel, cv2.COLOR_GRAY2RGB)
    Image.fromarray(skel_img.astype("uint8"), "RGB").show()

    shapes = list()
    visited = np.zeros((skel.shape[0], skel.shape[1]))
    for c in range(skel.shape[1]):
        for r in range(skel.shape[0]):
            if np.any(skel[r, c] < 100) and visited[r, c] == 0:
                shape = [(r, c), list(), list()]
                queue = [(r, c)]
                verts = 0
                points = [(r, c)]
                while len(queue) > 0:
                    current = queue[-1]
                    queue.remove(current)
                    visited[current[0], current[1]] = 1
                    points.append(current)
                    neighbors = find_neighbors(current, skel, visited)
                    queue.extend(neighbors[0])
                    if neighbors[1] == 1:
                        pruned = False
                        for n in find_neighbors(current, skel, visited, False)[0]:
                            if find_neighbors(n, skel, visited)[1] > 2:
                                pruned = True
                        if verts == 0 and not pruned:
                            shape[0] = current
                            verts += 1
                if verts == 0:
                    maximum = points[0]
                    for p in points:
                        if p[0] < maximum[0]:
                            maximum = p
                    shape[0] = maximum
                if len(points) > 0:
                    shapes.append(shape)

    visited = np.zeros((skel.shape[0], skel.shape[1]))
    for shape in shapes:
        e = shape[0]
        vertices = list()
        loops = list()
        n = find_neighbors(e, skel, visited)
        queue = n[0]
        current_vertex = [[e], n[1], 0, list()]
        vertices.append(current_vertex)
        vertex_queue = list()
        branch = list()
        visited[e[0], e[1]] = 1
        while len(queue) > 0:
            current = queue[-1]
            queue.remove(current)
            if visited[current[0], current[1]] == 0 or any([current in v[0] for v in vertices]):
                """if len(branch) % 5 == 0 and shapes.index(shape) > 12:
                    display = cv2.cvtColor(skel, cv2.COLOR_GRAY2RGB)
                    display[current[0], current[1]] = (0, 255, 0)
                    for b in current_vertex[3]:
                        for p in b:
                            display[p[0], p[1]] = (0, 0, 255)
                    display[current_vertex[0][0][0],
                            current_vertex[0][0][1]] = (255, 0, 0)
                    #Image.fromarray(display.astype("uint8"), "RGB").show()
                    # time.sleep(1)"""

                """for l in loops:
                    if len(l[0]) > 1:
                        print("===")
                        print(l[0][0][3])
                        print(l[0][1][3])
                        print("===")"""

                visited[current[0], current[1]] = 1
                neighbors = find_neighbors(current, skel, visited)
                all_neigh = find_neighbors(current, skel, visited, False)[0]
                if neighbors[1] > 2:
                    # junction
                    merged = False
                    merged_self = False
                    loop = False
                    vertices_merged = list()
                    for v in vertices:
                        if v[1] == 1:
                            continue
                        for p in v[0]:
                            if p in all_neigh:
                                if v == current_vertex and len(branch) > 2:
                                    #print("type 1")
                                    loops.append(
                                        [[current_vertex], [branch], 0])
                                    loop = True
                                elif v == current_vertex:
                                    merged_self = True
                                elif v in vertex_queue and len(branch) > 0 and len(v[3]) > 0:
                                    #print("type 2")
                                    b = list(reversed(v[3][0]))
                                    # print(v[0])
                                    # print(len(current_vertex[3]))
                                    x = [math.inf, 0]
                                    y = [math.inf, 0]
                                    for p in b + branch:
                                        if p[1] < x[0]:
                                            x[0] = p[1]
                                        if p[1] > x[1]:
                                            x[1] = p[1]
                                        if p[0] < y[0]:
                                            y[0] = p[0]
                                        if p[0] > y[1]:
                                            y[1] = p[0]
                                    if b in current_vertex[3] and (x[1]-x[0])*2 > y[1]-y[0] and current_vertex[1] != 2:
                                        current_vertex[3].remove(b)
                                        v[3] = list()
                                        loops.append(
                                            [[current_vertex, v], [branch, b], 0])
                                        loop = True
                                    else:
                                        #print("not loop")
                                        current_vertex[3].append(branch)
                                        v[3].append(list(reversed(branch)))
                                elif v in vertex_queue and len(branch) == 0 and len(current_vertex[3]) > 0 and len(v[3]) > 0:
                                    #print("type1 *")
                                    b = list(reversed(v[3][0]))
                                    if b in current_vertex[3]:
                                        current_vertex[3].remove(b)
                                        vertices.remove(v)
                                        vertex_queue.remove(v)
                                        current_vertex[0].extend(v[0])
                                        for p in current_vertex[0]:
                                            queue.extend(find_neighbors(
                                                p, skel, visited)[0])
                                        loops.append(
                                            [[current_vertex], [b], 0])
                                        loop = True
                                if current not in v[0]:
                                    v[0].append(current)
                                if v not in vertices_merged:
                                    vertices_merged.append(v)
                                branch = list()
                                # print("merged")
                                merged = True
                    if len(vertices_merged) == 2 and not loop:
                        #print("vert merged")
                        vertices.remove(vertices_merged[1])
                        if vertices_merged[1] in vertex_queue:
                            vertex_queue.remove(vertices_merged[1])
                        vertices_merged[0][0].extend(vertices_merged[1][0])
                        vertices_merged[0][3].extend(vertices_merged[1][3])
                        for b1 in vertices_merged[0][3]:
                            for b2 in vertices_merged[0][3]:
                                if b1 == b2:
                                    continue
                                for v in vertices:
                                    if v in vertices_merged:
                                        continue
                                    if (b1 in v[3] or list(reversed(b1)) in v[3]) and (b2 in v[3] or list(reversed(b2)) in v[3]):
                                        #print("found loop")
                                        v[3].remove(b1 if b1 in v[3]
                                                    else list(reversed(b1)))
                                        v[3].remove(b2 if b2 in v[3]
                                                    else list(reversed(b2)))
                                        # print(vertices_merged[0][3])
                                        # print(b1)
                                        vertices_merged[0][3].remove(b1)
                                        vertices_merged[0][3].remove(b2)
                                        loops.append(
                                            [[v, vertices_merged[0]], [b1, b2], 0])
                        current_vertex = vertices_merged[0]
                        branch = list()
                        for p in current_vertex[0]:
                            queue.extend(find_neighbors(p, skel, visited)[0])

                    if merged_self:
                        queue.extend(neighbors[0])
                        # print("self")
                    elif not merged and len(branch) == 1 and current_vertex[1] != 1:
                        current_vertex[0].append(current)
                        current_vertex[0].append(branch[0])
                        branch = list()
                        queue.extend(neighbors[0])
                    elif not merged and all([current not in v[0] for v in vertices]):
                        #print("junction - " + str(current))
                        vertex = [[current],
                                  neighbors[1], 0, [list(reversed(branch))]]
                        current_vertex[3].append(branch)
                        for i in range(len(loops)):
                            if any(v in current_vertex[0] for v in loops[i][0][0][0]):
                                # print("changed")
                                loops[i][0][0] = current_vertex
                            # print(current_vertex[0])
                            if len(loops[i][0]) > 1 and any(v in current_vertex[0] for v in loops[i][0][1][0]):
                                # print("changed2")
                                loops[i][0][1] = current_vertex
                                # print(len(loops[i][0][1][3]))
                                # print(i)

                        branch = list()
                        vertices.append(vertex)
                        vertex_queue.append(vertex)
                elif neighbors[1] == 0:
                    vertices.append([current], 0, 0, [])
                elif neighbors[1] == 1:
                    # endpoint
                    if len(branch) > 0 or current_vertex[1] == 1:
                        vertices.append(
                            [[current], 1, 0, [list(reversed(branch))]])
                        current_vertex[3].append(branch)
                        branch = list()
                    else:
                        skel[current[0], current[1]] = 255
                        for k in connectivity_kernels:
                            c = cv2.morphologyEx(255-skel,
                                                 cv2.MORPH_HITMISS, k)
                            skel = np.array((skel + c), np.uint8)
                        hasLoop = False
                        for l in loops:
                            if current_vertex in l[0]:
                                hasLoop = True
                                break
                        if current_vertex[1] > 3 or hasLoop:
                            current_vertex[1] -= 1
                        else:
                            # print("prune")
                            # print(len(current_vertex[3]))
                            vertices.remove(current_vertex)
                            if len(current_vertex[3]) == 0:
                                for l in loops:
                                    if current_vertex in l[0]:
                                        if len(l[0]) == 2:
                                            l[0].remove(current_vertex)
                                            l[1][0].extend(current_vertex[0])
                                            l[1][0].extend(
                                                list(reversed(l[1][1])))
                                            l[1].remove(l[1][1])
                            elif len(current_vertex[3]) == 1:
                                for v in vertices:
                                    for b in v[3]:
                                        if b == current_vertex[3][0] or b == list(reversed(current_vertex[3][0])):
                                            branch = b
                                            v[3].remove(branch)
                                            branch.extend(current_vertex[0])
                                            for p in current_vertex[0]:
                                                queue.extend(find_neighbors(
                                                    p, skel, visited)[0])
                                            # print(len(queue))
                                            current_vertex = v
                            else:
                                new_vertex = vertices[0]
                                for v in vertices:
                                    for b in v[3]:
                                        if b == list(reversed(current_vertex[3][0])):
                                            # print("yo1")
                                            # print(current_vertex[3])
                                            v[3].remove(b)
                                            branch = b
                                            # branch.extend(current_vertex[0])
                                            # branch.extend(current_vertex[3][1])
                                            new_vertex = v
                                            # print(v[3])
                                    for b in v[3]:
                                        if b == list(reversed(current_vertex[3][1])):
                                            if len(v[3]) > 1:
                                                v[3][v[3].index(b)] = b + list(
                                                    reversed(branch))
                                                new_vertex[3].append(
                                                    branch + list(
                                                        reversed(b)))
                                                branch = list()
                                            else:
                                                # print("yo2")
                                                if v in vertex_queue:
                                                    vertex_queue.remove(v)
                                                vertices.remove(v)
                                                visited[v[0][0][0],
                                                        v[0][0][1]] = 0
                                                for p in current_vertex[0]:
                                                    visited[p[0], p[1]] = 0
                                                for p in current_vertex[3][1]:
                                                    visited[p[0], p[1]] = 0
                                                queue.clear()
                                                queue.extend(
                                                    find_neighbors(current_vertex[3][0][0], skel, visited)[0])
                                            # print(len(queue))
                                current_vertex = new_vertex
                            # print(current_vertex)
                            display = cv2.cvtColor(
                                skel, cv2.COLOR_GRAY2RGB)
                            display[current[0], current[1]] = (0, 255, 0)
                            for b in current_vertex[3]:
                                for p in b:
                                    display[p[0], p[1]] = (0, 0, 255)
                            for p in branch:
                                display[p[0], p[1]] = (0, 255, 255)
                            display[current_vertex[0][0][0],
                                    current_vertex[0][0][1]] = (255, 0, 0)
                            #Image.fromarray(display.astype("uint8"), "RGB").show()
                else:
                    # line
                    loop = False
                    branch.append(current)
                    for p in current_vertex[0]:
                        if p in all_neigh and len(branch) > 2:
                            #print("type 1 -")
                            loops.append([[current_vertex], [branch], 0])
                            branch = list()
                            loop = True
                            break
                    for v in vertex_queue:
                        if len(v[3]) == 0:
                            break
                        for p in v[0]:
                            if p in all_neigh and len(branch) > 0:
                                if len(branch) == 1:
                                    if v in vertices and list(reversed(v[3][0])) in current_vertex[3]:
                                        vertices.remove(v)
                                        vertex_queue.remove(v)
                                        current_vertex[0].extend(v[0])
                                        current_vertex[0].append(current)
                                        #print("type 1 --")
                                        current_vertex[3].remove(
                                            list(reversed(v[3][0])))
                                        loops.append(
                                            [[current_vertex], [v[3][0] + list(reversed(branch))], 0])
                                        branch = list()
                                        for p in current_vertex[0]:
                                            queue.extend(find_neighbors(
                                                p, skel, visited)[0])
                                        loop = True
                                        break
                                elif len(branch) > 2:
                                    #print("type 2 -")
                                    b = list(reversed(v[3][0]))
                                    x = [math.inf, 0]
                                    y = [math.inf, 0]
                                    for p in b + branch:
                                        if p[1] < x[0]:
                                            x[0] = p[1]
                                        if p[1] > x[1]:
                                            x[1] = p[1]
                                        if p[0] < y[0]:
                                            y[0] = p[0]
                                        if p[0] > y[1]:
                                            y[1] = p[0]
                                    if b in current_vertex[3] and (x[1]-x[0])*2 > y[1]-y[0] and current_vertex[1] != 2:
                                        current_vertex[3].remove(b)
                                        v[3] = list()
                                        loops.append([[current_vertex, v],
                                                      [branch, b], 0])
                                        branch = list()
                                        loop = True
                                    else:
                                        #print("not loop")
                                        current_vertex[3].append(branch)
                                        v[3].append(list(reversed(branch)))
                                    break

                    if not loop:
                        queue.extend(neighbors[0])
            while len(queue) == 0 and len(vertex_queue) > 0:
                current_vertex = vertex_queue[-1]
                # print(len(current_vertex[3]))
                vertex_queue.remove(current_vertex)
                for p in current_vertex[0]:
                    queue.extend(find_neighbors(p, skel, visited)[0])
                # print("searched point " +
                    # str(current_vertex[0][current_vertex[2]]))
                #print("extended queue: " + str(len(queue)))
                current_vertex[2] += 1
                branch = list()

        shape[1] = vertices
        shape[2] = loops

    #print("found vertices")

    verts = cv2.cvtColor(skel, cv2.COLOR_GRAY2RGB)
    for s in shapes:
        for v in s[1]:
            # if len(v[3]) == 0:
            #   s[1].remove (v)
            v_neighbors = list()
            for p in v[0]:
                all_neigh_p = find_neighbors(
                    p, skel, visited, use_visited=False)[0]
                for n in all_neigh_p:
                    if n not in v[0] and n not in v_neighbors:
                        v_neighbors.append(n)
            v[1] = len(v_neighbors)
            if v[1] == 0:
                color = (255, 255, 0)
            elif v[1] == 1:
                color = (0, 255, 0)
            elif v[1] == 3:
                color = (0, 0, 255)
            elif v[1] == 4:
                color = (255, 0, 0)
            else:
                color = (255, 0, 255)
            totaly = 0
            totalx = 0
            for p in v[0]:
                totaly += p[0]
                totalx += p[1]
                verts[p[0], p[1]] = color
            v[0] = (round(totaly / len(v[0])), round(totalx / len(v[0])))
            v[2] = 0

    #Image.fromarray(verts.astype("uint8"), "RGB").show()

    for s in shapes:
        for l in s[2]:
            color = (random.randint(0, 1)*255, random.randint(0, 1)
                     * 255, random.randint(0, 1)*255)
            if color == (255, 255, 255) or color == (0, 0, 0):
                color = (255, 0, 0)
            if len(l[0]) > 1:
                if len(l[0][0][3]) == 0 and l[0][0][1] == 2:
                    if distance(l[1][0][-1], l[1][1][0]) < distance(l[1][0][-1], l[1][1][-1]):
                        combined_branch = l[1][0] + l[1][1]
                    else:
                        combined_branch = l[1][0] + list(reversed(l[1][1]))
                    s[1].remove(l[0][0])
                    s[2][s[2].index(l)] = [[l[0][1]], [combined_branch], 0]
                elif len(l[0][1][3]) == 0 and l[0][1][1] == 2:
                    if distance(l[1][0][-1], l[1][1][0]) < distance(l[1][0][-1], l[1][1][-1]):
                        combined_branch = l[1][0] + l[1][1]
                    else:
                        combined_branch = l[1][0] + list(reversed(l[1][1]))
                    s[1].remove(l[0][1])
                    s[2][s[2].index(l)] = [[l[0][0]], [combined_branch], 0]
            if type(l[0][0][0]) == list:
                l[0][0][0] = l[0][0][0][0]
            if len(l[0]) > 1 and type(l[0][1][0]) == list:
                l[0][1][0] = l[0][1][0][0]
            for e in l[1]:
                for p in e:
                    verts[p[0], p[1]] = (255, 0, 0) if len(
                        l[1]) > 1 else (0, 0, 255)

    #Image.fromarray(verts.astype("uint8"), "RGB").show()

    paths = list()
    skel = cv2.cvtColor(skel, cv2.COLOR_GRAY2RGB)
    path_im = np.full(skel.shape, 255)
    for shape in shapes:
        path = list()
        vertices = shape[1]
        loops = shape[2]

        if len(vertices) == 0:
            continue

        if len(vertices) == 1 and len(loops) == 0:
            #print("single:" + str(vertices[0][0]))
            paths.append([vertices[0][0]])
            continue

        #print("shape:" + str(len(vertices)))
        # print(len(loops))

        path_im = np.copy(skel)
        current, previous_edge, vertices = find_start(vertices, loops)
        if current == None:
            continue
        visited = 0
        double_traced = False
        while visited < len(vertices):
            # print("---")
            # print(visited)
            # print(current)
            hasLoop = False
            pot_loops = list()
            followed_loop = False
            sharp_angle = False
            for l in loops:
                if current in l[0]:
                    hasLoop = True
                    pot_loops.append(l)
            loop = None
            if len(pot_loops) > 0:
                loop = pot_loops[0]
                for l in pot_loops:
                    if l[2] < loop[2]:
                        loop = l

            if shapes.index(shape) > 20:
                path_im = np.copy(skel)
                for p in path:
                    path_im[p[0], p[1]] = (0, 0, 255)
                path_im[current[0][0], current[0][1]] = (255, 0, 0)
                #Image.fromarray(path_im.astype("uint8"), "RGB").show()
            # time.sleep(1)
            path.append(current[0])
            current[2] += 1
            if current[2] == (1 if len(current[3]) == 1 else 2) and (not hasLoop or all(l[2] > 0 for l in pot_loops if l != loop)):
                visited += 1
                if visited == len(vertices) and not hasLoop:
                    break
            if len(loops) == 0 and len(current[3][0]) == 0:
                #print("no branches")
                current, previous_edge = find_start(vertices, loops)
                continue
            elif hasLoop and loop[2] == 0:
                followed_loop = True
                # print("loop")
                # follow loop
                loop_path = follow_loop(previous_edge, loop)
                loop[2] += 1
                previous_edge = loop_path[0]
                path.extend(loop_path[0])
                current = loop_path[1]
                double_traced = False
            elif hasLoop and loop[2] > 0 and len(loop[0]) > 1 and all(b[0] in path for b in current[3]) \
                and (collinear(loop[0][0][0], loop[1][0][int(len(loop[1][0]) / 2)], loop[0][1][0]) or
                     collinear(loop[0][0][0], loop[1][1][int(len(loop[1][1]) / 2)], loop[0][1][0])):
                #print("retraced loop")
                followed_loop = True
                if collinear(loop[0][0][0], loop[1][0][int(len(loop[1][0]) / 2)], loop[0][1][0]):
                    if current == loop[0][0]:
                        previous_edge = loop[1][0]
                        path.extend(loop[1][0])
                        current = loop[0][1]
                    else:
                        previous_edge = list(reversed(loop[1][0]))
                        path.extend(list(reversed(loop[1][0])))
                        current = loop[0][0]
                else:
                    if current == loop[0][0]:
                        previous_edge = loop[1][1]
                        path.extend(loop[1][1])
                        current = loop[0][1]
                    else:
                        previous_edge = list(reversed(loop[1][1]))
                        path.extend(list(reversed(loop[1][1])))
                        current = loop[0][0]
                double_traced = False
            elif len(current[3]) == 1:
                #print("only path")
                # choose only path
                b = current[3][0]
                if previous_edge != None and (b == previous_edge or b == list(reversed(previous_edge))):
                    double_traced = True
                else:
                    double_traced = False
                previous_edge = b
                if visited == len(vertices) and b[0] in path and b[-1] in path:
                    break
                else:
                    path.extend(b)
            elif len(current[3]) == 3 and current[2] == 2:
                #print("untraced path")
                # choose untraced path
                if current[3][0][0] not in path and all((current[3][0][0] not in p) for p in paths):
                    b = current[3][0]
                    # print("b1")
                elif current[3][1][0] not in path and all((current[3][1][0] not in p) for p in paths):
                    b = current[3][1]
                    # print("b2")
                elif current[3][2][0] not in path and all((current[3][2][0] not in p) for p in paths):
                    b = current[3][2]
                    # print("b3")
                else:
                    closest_unvisited = None
                    shortest_dist = math.inf
                    for v in vertices:
                        dist = distance(v[0], current[0])
                        if v[2] < (2 if len(v[3]) > 1 else 1) and (closest_unvisited == None or dist < shortest_dist):
                            closest_unvisited = v
                            shortest_dist = dist
                    if closest_unvisited == None:
                        if previous_edge != None:
                            b = previous_edge
                        # print("retrace")
                    else:
                        best_b = current[3][0]
                        for B in current[3]:
                            if distance(B[-1], closest_unvisited[0]) < distance(best_b[-1], closest_unvisited[0]):
                                best_b = B
                        b = best_b
                if previous_edge == None or b == previous_edge:
                    previous_edge = b
                    path.extend(b)
                else:
                    # print(getSmoothness(previous_edge, current[0], b))
                    s = get_true_angle(
                        previous_edge[0], current[0], b[-1]) < 110
                    # print(s)
                    # print(has_straight_line(current))
                    # print(double_traced)
                    path_im = np.copy(skel)
                    # collinear(current[0], b[int(len(b) / 2)], b[-1])
                    if min(len(previous_edge), len(b)) > 5 and collinear(current[0], b[int(len(b) / 2)], b[-1]) and has_straight_line(current) and s and current[2] > 1 \
                            and (not collinear(previous_edge[0], current[0], b[-1]) or min(len(b), len(previous_edge)) < 5) and double_traced and len(previous_edge) > 2:
                        sharp_angle = True
                        # print("sharp")
                        path_im = np.copy(skel)
                        for p in previous_edge:
                            path_im[p[0], p[1]] = (0, 0, 255)
                        for p in b:
                            path_im[p[0], p[1]] = (0, 255, 0)
                        # Image.fromarray(path_im.astype("uint8"), "RGB").show()
                    else:
                        previous_edge = b
                        path.extend(b)
                double_traced = False
            elif (current[2] == 1 or current[1] >= 4) and len(current[3]) > 0:
                #print("smoothest path")
                # choose smoothest path
                # print(len(path))
                max_smooth = math.inf
                smooth_branch = current[3][0]
                short_branch = None
                if previous_edge != None and len(previous_edge) > 0:
                    # print(len(current[3]))
                    for b in current[3]:
                        if len(b) < 1 or b == previous_edge:
                            continue
                        smoothness = getSmoothness(
                            previous_edge, current[0], b)  # - (current[0][0] - getHight(b, 3)) / 6

                        # print((current[0][0] - getHight(b)) / 5)
                        # print(smoothness)
                        if smoothness < max_smooth and b[0] not in path:
                            max_smooth = smoothness
                            smooth_branch = b
                        if (short_branch == None or len(b) < len(short_branch)) and b[0] not in path:
                            short_branch = b
                use_smooth = True
                """if short_branch != None and len(short_branch) < len(smooth_branch) / 4 and len(short_branch) < 4:
                    use_smooth = False
                    if previous_edge != None:
                        for v in vertices:
                            if v != current and (short_branch in v[3] or list(reversed(short_branch)) in v[3]):
                                if v[1] > 1:
                                    use_smooth = True
                                break"""
                if use_smooth:
                    if previous_edge != None:
                        s = (getSmoothness(
                            previous_edge, current[0], smooth_branch) > 1.5) if min(len(previous_edge), len(smooth_branch)) < 6 \
                            else get_true_angle(previous_edge[0], current[0], smooth_branch[-1]) < 110
                        # print(has_straight_line(current))
                        # print(s)
                        # print(double_traced)
                        collinear(current[0], smooth_branch[int(
                            len(smooth_branch) / 2)], smooth_branch[-1])
                        if has_straight_line(current) and s \
                                and current[2] > 1 and (not collinear(previous_edge[0], current[0], smooth_branch[-1]) or min(len(smooth_branch), len(previous_edge)) < 5) and double_traced and len(previous_edge) > 2:
                            sharp_angle = True
                            # print("sharp")
                        else:
                            previous_edge = smooth_branch
                            path.extend(smooth_branch)
                    else:
                        previous_edge = smooth_branch
                        path.extend(smooth_branch)
                """else:
                    print("short")
                    previous_edge = short_branch
                    path.extend(short_branch)"""
                # print(len(path))
                double_traced = False
            if not followed_loop and not sharp_angle:
                for v in vertices:
                    if v != current and (previous_edge in v[3] or list(reversed(previous_edge)) in v[3]):
                        current = v
                        break
            if (current[1] == 1 and current[2] == 2) or (current[1] == 3 and current[2] == 3) or (hasLoop and loop[2] == 2) or sharp_angle or current[2] > 4:
                #print("new shape")
                if double_traced or sharp_angle:
                    path.reverse()
                    for p in previous_edge:
                        path.remove(p)
                    path.remove(current[0])
                    path.reverse()
                    if sharp_angle:
                        if current[2] == 2:
                            visited -= 1
                        current[2] -= 1
                path_im = np.copy(skel)
                current, previous_edge, vertices = find_start(
                    vertices, loops)
                paths.append(path)
                path = list()
                double_traced = False
                if current == None:
                    break
        if len(path) > 0:
            paths.append(path)

    #print("found paths")

    total_smoothness = 0
    point_count = 0
    for path in paths:
        for i in range(5, len(path)-5):
            total_smoothness += getSmoothness(path[: i], path[i], path[i:], 10)
            point_count += 1
    average_smoothness = total_smoothness/point_count

    strokes = list()
    path_im = np.full(skel.shape, 255)
    previous = (0, 0)
    delta = 1000 / skel.shape[1]
    for path in paths:
        for i in range(len(path)):
            if all(path_im[path[i][0], path[i][1]]) == 0:
                path_im[path[i][0], path[i][1]] = (0, 0, 255)
            elif any(path_im[path[i][0], path[i][1]]) == 0:
                path_im[path[i][0], path[i][1]] = (255, 0, 0)
            else:
                path_im[path[i][0], path[i][1]] = (0, 0, 0)
            if i > 0 and i < len(path)-1:
                smooth = getSmoothness(
                    path[: i], path[i], path[i:], 10) / average_smoothness
                if smooth == 0:
                    smooth = 1
            else:
                smooth = 1
            if i % 5 == 0 or i % min(5, math.ceil(5 / smooth)) == 0 or i == len(path)-1:
                posx = float(path[i][1])
                posy = line_height - float(path[i][0])
                if i == 0:
                    path_im[path[i][0], path[i][1]] = (0, 255, 0)
                if i == len(path)-1:
                    strokes.append((posx, posy, 1))
                else:
                    strokes.append((posx, posy, 0))
                #if i != 0:
                    #window.after(100, draw_line(
                    #    previous[1], 200 - (previous[0] + line_height), posx, 200 - (posy + line_height)))
                previous = (posy, posx)
            # if count % 15 == 0:
            #    Image.fromarray(path_im.astype("uint8"), "RGB").show()

    strokes = np.array(strokes)

    #Image.fromarray(path_im.astype("uint8"), "RGB").show()

    return strokes


def is_right(vertex):
    count = 0
    for b1 in vertex[3]:
        for b2 in vertex[3]:
            if b1 == b2 or len(b1) < 5 or len(b2) < 5:
                continue
            # print(get_true_angle(b1[-1], vertex[0], b2[-1]))
            if abs(get_true_angle(b1[-1], vertex[0], b2[-1]) - 90) < 10 \
                    and abs(get_true_angle(b1[int(len(b1) / 2)], vertex[0], b2[int(len(b2) / 2)]) - 90) < 10:
                count += 1
    return count > 1


def has_straight_line(vertex):
    for b1 in vertex[3]:
        for b2 in vertex[3]:
            if b1 == b2 or len(b1) == 0 or len(b2) == 0:
                continue
            if distance(b1[0], b2[0]) < distance(b1[-1], b2[0]):
                b1 = list(reversed(b1))
            #print(getSmoothness(b1, vertex[0], b2, 8))
            if getSmoothness(b1, vertex[0], b2, 8) < 1:
                return True
    return False


def get_true_angle(p1, v, p2):
    a = distance(v, p2)
    b = distance(p1, p2)
    c = distance(v, p1)
    if a == 0 or b == 0 or c == 0:
        return 0
    return math.degrees(math.acos(max(-1, min(1, ((math.pow(b, 2) - math.pow(a, 2) - math.pow(c, 2)) / (-2*a*b))))))


def find_start(vertices, loops):
    topX = math.inf
    topY = math.inf
    for v in vertices:
        for b in v[3]:
            for p in b:
                if p[0] < topY:
                    topY = p[0]
                if p[1] < topX:
                    topX = p[1]
    start_vertex = None
    previous_edge = None
    min_dist = math.inf
    for v in vertices:
        dist = abs(v[0][1]-topX) + abs(v[0][0] - topY)*2
        if ((v[1] == 1 and len(v[3]) == 1 and v[2] == 0) or (v[1] - v[2] == 2 and len(v[3]) - v[2] == 2)) and len(v[3]) != 4 and dist < min_dist:
            taller_vert = False
            for v2 in vertices:
                if v != v2 and v2[2] == 0 and v2[1] > 1 and v2[0][0] < v[0][0]:
                    taller_vert = True
            if not taller_vert:
                start_vertex = v
                min_dist = dist
    if len(vertices) > 2:
        best_p = None
        best_b = None
        for v in vertices:
            for b in v[3]:
                for p in range(5, len(b)-5):
                    smooth = get_edge_smoothness(b, p)
                    angle = get_true_angle(b[0], b[p], b[-1])
                    dist = abs(b[p][1] - topX) + abs(b[p][0] - topY) * 2
                    if smooth > 1 and angle < 90 and len(b) - p > 10 and p > 10 and dist < min_dist:
                        min_dist = dist
                        best_p = p
                        best_b = b
        if best_p != None:
            new_b1 = best_b[:best_p]
            new_b2 = best_b[best_p+1:]

            for v in vertices:
                for b in v[3]:
                    if b == best_b:
                        v[3][v[3].index(b)] = new_b1
                        # print("hi1")
                    if b == list(reversed(best_b)):
                        # print("hi2")
                        v[3][v[3].index(b)] = list(reversed(new_b2))

            new_v1 = [best_b[best_p], 1, 0, [list(reversed(new_b1))]]
            new_v2 = [best_b[best_p], 1, 0, [new_b2]]
            vertices.append(new_v1)
            vertices.append(new_v2)
            start_vertex = new_v1

    if start_vertex == None:
        # print("None")
        start_vertex = None
        start_x = math.inf
    else:
        start_x = start_vertex[0][1]
    for l in loops:
        total_x = 0
        total_y = 0
        total_points = 0
        for e in l[1]:
            for p in e:
                total_x += p[1]
                total_y += p[0]
                total_points += 1
        x = total_x / total_points
        y = total_y / total_points
        dist = abs(x-topX) + abs(y - topY)*2
        if (x < start_x or dist < min_dist) and l[0][0][2] == 0 and l[2] == 0:
            if len(l[0]) == 1 or l[0][0][0][0] < l[0][1][0][0]:
                new_start_vertex = l[0][0]
            else:
                new_start_vertex = l[0][1]
            start_x = x
            min_dist = dist
            start_vertex = new_start_vertex
    for v in vertices:
        if v[1] == 1 and v[2] == 0 and len(v[3]) > 0 and len(v[3][0]) > 10 and len(vertices) > 2:
            x = [math.inf, 0]
            y = [math.inf, 0]
            x_total = 0
            y_total = 0
            for p in v[3][0]:
                if p[1] < x[0]:
                    x[0] = p[1]
                if p[1] > x[1]:
                    x[1] = p[1]
                if p[0] < y[0]:
                    y[0] = p[0]
                if p[0] > y[1]:
                    y[1] = p[0]
                x_total += p[1]
                y_total += p[0]
            overlap_average = 0
            for p in v[3][0]:
                previous_overlap = False
                for r in range(y[0], y[1]+1):
                    if (r, p[1]) in v[3][0]:
                        if not previous_overlap:
                            overlap_average += 1
                            previous_overlap = True
                    else:
                        previous_overlap = False
            overlap_average /= len(v[3][0])
            x_center = x_total / len(v[3][0])
            y_center = y_total / len(v[3][0])
            dist = abs(x_center - topX) + abs(y_center - topY) * 2
            if y[1] - y[0] <= 0 or x[1] - x[0] <= 0:
                continue
            #print("overlap " + str(overlap_average))
            if len(v[3][0]) > 10 and abs((x[1] - x[0]) / (y[1] - y[0])) > 0.5 and overlap_average > 1 and overlap_average < 2 \
                    and (x_center < start_x or dist < min_dist):
                start_x = x_center
                min_dist = dist
                start_vertex = v

    # print(start_vertex)
    return start_vertex, previous_edge, vertices


def follow_loop(edge, loop):
    path = list()
    exit_ = loop[0][0]
    if len(loop[0]) == 1:
        if edge == None and len(loop[0][0][3]) == 1:
            if len(loop[0][0][3][0]) == 0 or getSmoothness(loop[1][0], loop[0][0][0], loop[0][0][3][0]) < getSmoothness(list(reversed(loop[1][0])), loop[0][0][0], loop[0][0][3][0]):
                path.extend(loop[1][0])
            else:
                path.extend(list(reversed(loop[1][0])))
        else:
            if getHight(loop[1][0]) < getHight(list(reversed(loop[1][0]))):
                path.extend(loop[1][0])
            else:
                path.extend(list(reversed(loop[1][0])))

    else:
        vertex = loop[0][0]
        if edge == None:
            if loop[0][1][0] < loop[0][0][0]:
                vertex = loop[0][1]
                loop[1][0] = list(reversed(loop[1][0]))
                loop[1][1] = list(reversed(loop[1][1]))
                exit_ = loop[0][1]
            if getHight(loop[1][0]) < getHight(loop[1][1]):
                path.extend(loop[1][0])
                if vertex == loop[0][0]:
                    path.append(loop[0][1][0])
                else:
                    path.append(loop[0][0][0])
                path.extend(list(reversed(loop[1][1])))
            else:
                path.extend(loop[1][1])
                if vertex == loop[0][0]:
                    path.append(loop[0][1][0])
                else:
                    path.append(loop[0][0][0])
                path.extend(list(reversed(loop[1][0])))
        else:
            if edge in loop[0][1][3] or list(reversed(edge)) in loop[0][1][3]:
                loop[1][0] = list(reversed(loop[1][0]))
                loop[1][1] = list(reversed(loop[1][1]))
                # print("loop 1 vertex")
                vertex = loop[0][1]
            else:
                exit_ = loop[0][1]
            edge1_visited = False
            edge2_visited = False
            # trace highest edge
            if getHight(loop[1][0]) < getHight(loop[1][1]):
                path.extend(loop[1][0])
                edge1_visited = True
            else:
                path.extend(loop[1][1])
                edge2_visited = True
            # backtrack in higher direction
            if getHight(list(reversed(loop[1][0]))) < getHight(list(reversed(loop[1][1]))):
                path.extend(list(reversed(loop[1][0])))
                if edge1_visited:
                    path.extend(loop[1][1])
                elif len(loop[1][0]) < len(loop[1][1]):
                    path.extend(loop[1][0])
                else:
                    path.extend(loop[1][1])
            else:
                path.extend(list(reversed(loop[1][1])))
                if edge2_visited:
                    path.extend(loop[1][0])
                elif len(loop[1][0]) < len(loop[1][1]):
                    path.extend(loop[1][0])
                else:
                    path.extend(loop[1][1])
    return path, exit_


def getHight(edge, length=5):
    total = 0
    for k in range(min(length, len(edge))):
        total += edge[k][0]
    return total / (length if length <= len(edge) else (length-len(edge)))


def get_edge_smoothness(edge, vertex_ind, chord_length=5):
    total_d = 0
    for k in range(0, chord_length):

        c1 = vertex_ind-(chord_length-k)
        if c1 < 0:
            c1 = len(edge) + vertex_ind - (chord_length - k)
        c2 = vertex_ind + k
        if c2 > len(edge)-1:
            c2 = vertex_ind + k - len(edge)
        chord = (edge[c1], edge[c2])

        if chord[1][1] - chord[0][1] == 0:
            d = abs(chord[1][1] - edge[vertex_ind][1])
        elif chord[1][0] - chord[0][0] == 0:
            d = abs(chord[1][0] - edge[vertex_ind][0])
        else:
            slope = (chord[1][0] - chord[0][0]) / (chord[1][1] - chord[0][1])
            a = slope
            b = -1
            c = chord[0][0] + slope * -chord[0][1]
            d = (abs(a * edge[vertex_ind][1] + b * edge[vertex_ind]
                     [0] + c)) / (math.sqrt(pow(a, 2) + pow(b, 2)))
        total_d += d
    return total_d / chord_length


def getSmoothness(edge1, vertex, edge2, chord_length=5):
    edge1 = list(edge1)
    edge2 = list(edge2)
    if len(edge1) < chord_length:
        if edge1[0][1] - edge1[-1][1] == 0:
            # print("1y")
            endy = (edge1[0][0] + (chord_length - len(
                edge1))) if edge1[0][0] > edge1[-1][0] else (edge1[0][0] - (chord_length - len(edge1)))
            # print(edge1[0][0])
            # print(endy)
            step = -1 if endy < edge1[0][0] else 1
            # print(step)
            # print([(y, edge1[0][1]) for y in range(
            #    edge1[0][0] + step, endy+step, step)])
            edge1.extend([(y, edge1[0][1]) for y in range(
                edge1[0][0] + step, endy + step, step)])
        elif edge1[0][0] - edge1[-1][0] == 0:
            # print("1x")
            endx = (edge1[0][1] + (chord_length - len(
                edge1))) if edge1[0][1] > edge1[-1][1] else (edge1[0][1] - (chord_length - len(edge1)))
            # print(edge1[0][1])
            # print(endx)
            step = -1 if endx < edge1[0][1] else 1
            # print(step)
            # print([(edge1[0][0], x) for x in range(
            #    edge1[0][1] + step, endx+step, step)])
            edge1.extend([(edge1[0][0], x) for x in range(
                edge1[0][1] + step, endx+step, step)])
        else:
            slope = (edge1[0][0]-edge1[-1][0]) / (edge1[0][1]-edge1[-1][1])
            b = -slope * edge1[0][1] + edge1[0][0]
            if abs(slope) > 1:
                newx = round((edge1[0][0] - b) / slope)
                newy = round((edge1[0][0] + (chord_length-len(edge1))))
                edge1.extend(connect(edge1[0], (newy, newx)))
                # print(edge1[0])
                # print(str((newy, newx)))
                # print(len(edge1))
            else:
                newx = round(edge1[0][1] + (chord_length - len(edge1)))
                newy = round(slope*edge1[0][1] + b)
                edge1.extend(connect(edge1[0], (newy, newx)))
                # print(connect(edge1[0], (newy, newx)))
    if len(edge2) < chord_length:
        if edge2[-1][1] - edge2[0][1] == 0:
            # print("2y")
            endy = (edge2[-1][0] + (chord_length - len(
                edge2))) if edge2[-1][0] > edge2[0][0] else (edge2[-1][0] - (chord_length - len(edge2)))
            # print(edge2[-1][0])
            # print(endy)
            step = -1 if endy < edge2[-1][0] else 1
            # print(step)
            # print([range(edge2[-1][0] + step, endy + step, step)])
            edge2.extend([(y, edge2[-1][1]) for y in range(
                edge2[-1][0] + step, endy+step, step)])
        elif edge2[-1][0] - edge2[0][0] == 0:
            # print("2x")
            endx = (edge2[-1][1] + (chord_length - len(
                edge2))) if edge2[-1][1] > edge2[0][1] else (edge2[-1][1] - (chord_length - len(edge2)))
            # print(edge2[-1][1])
            # print(endx)
            step = -1 if endx < edge2[-1][1] else 1
            # print(step)
            # print([(edge2[-1][0], x) for x in range(
            # edge2[-1][1] + step, endx + step, step)])
            edge2.extend([(edge2[-1][0], x) for x in range(
                edge2[-1][1] + step, endx+step, step)])
        else:
            slope = (edge2[-1][0] - edge2[0][0]) / (edge2[-1][1] - edge2[0][1])
            b = -slope * edge2[-1][1] + edge2[-1][0]
            if abs(slope) > 1:
                newx = round((edge2[-1][0] - b) / slope)
                newy = round(edge2[-1][0] + (chord_length-len(edge2)))
                edge2.extend(connect(edge2[-1], (newy, newx)))
                # print(connect(edge2[-1], (newy, newx)))
            else:
                newx = round(edge2[-1][1] + (chord_length - len(edge2)))
                newy = round(slope*edge2[-1][1] + b)
                edge2.extend(connect(edge2[-1], (newy, newx)))
                # print(connect(edge2[-1], (newy, newx)))

    total_d = 0
    for k in range(0, chord_length):
        chord = (edge1[max(0, len(edge1)-(chord_length-k)-1)],
                 edge2[min(len(edge2)-1, k)])
        if chord[1][1] - chord[0][1] == 0:
            d = abs(chord[1][1] - vertex[1])
        else:
            slope = (chord[1][0] - chord[0][0]) / (chord[1][1] - chord[0][1])
            a = slope
            b = -1
            c = chord[0][0] + slope * -chord[0][1]
            d = (abs(a * vertex[1] + b * vertex
                     [0] + c)) / (math.sqrt(pow(a, 2) + pow(b, 2)))
        total_d += d
    return total_d / chord_length


def find_neighbors(pos, image, visited, use_visited=True):
    neighbors = list()
    count = 0
    for r in range(max(0, pos[0]-1), min(image.shape[0]-1, pos[0]+2)):
        for c in range(max(0, pos[1]-1), min(image.shape[1]-1, pos[1]+2)):
            if (r, c) != pos and np.any(image[r, c] < 100):
                if visited[r, c] == 0 or not use_visited:
                    neighbors.append((r, c))
                count += 1
    return neighbors, count


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


def get_contours(image):
    labels = np.zeros(image.shape)
    current_label = 1
    intensity_threshold = get_threshold(image)
    contours = list()
    for r in range(1, image.shape[0]-1):
        for c in range(1, image.shape[1]-1):
            is_contour = any(i > intensity_threshold for i in [image[r-1, c], image[r+1, c],
                                                               image[r, c-1], image[r, c+1]])
            if image[r][c] <= intensity_threshold and ((r, c) not in cont for cont in contours) and is_contour and labels[r][c] == 0:
                element = search_element(r, c, image, labels,
                                         intensity_threshold, current_label)[1]
                prev = (r - 1 if c - 1 < 0 else r, c if c - 1 < 0 else c - 1)
                contours.append(trace_contor(prev, (r, c), image, element))
                current_label += 1
    return contours


def fit_circle(p1, p2, p3, image):
    x12 = p1[1] - p2[1]
    x13 = p1[1] - p3[1]
    y12 = p1[0] - p2[0]
    y13 = p1[0] - p3[0]
    y31 = p3[0] - p1[0]
    y21 = p2[0] - p1[0]
    x31 = p3[1] - p1[1]
    x21 = p2[1] - p1[1]

    sx13 = pow(p1[1], 2) - pow(p3[1], 2)
    sy13 = pow(p1[0], 2) - pow(p3[0], 2)
    sx21 = pow(p2[1], 2) - pow(p1[1], 2)
    sy21 = pow(p2[0], 2) - pow(p1[0], 2)

    f = ((sx13) * (x12) + (sy13) * (x12) + (sx21) * (x13) +
         (sy21) * (x13)) / (2 * ((y31) * (x12) - (y21) * (x13)))
    g = ((sx13) * (y12) + (sy13) * (y12) + (sx21) * (y13) +
         (sy21) * (y13)) / (2 * ((x31) * (y12) - (x21) * (y13)))

    c = -pow(p1[1], 2) - pow(p1[0], 2) - 2 * g * p1[1] - 2 * f * p1[0]
    r = distance(p1, (-f, -g))
    """for a in range(0, 360, 30):
        angle = math.radians(a)
        image[min(max(int(-f + r * math.sin(angle)), 0), image.shape[0]-1),
              min(max(int(-g + r * math.cos(angle)), 0), image.shape[1]-1)] = (0, 255, 255)"""

    return (-f, -g)


def get_angle(p1, p2):
    return math.atan2(p2[0] - p1[0], p2[1] - p1[1])


def collinear(p1, p2, p3, tolerance=2):
    return distance(p1, p2) + distance(p2, p3) - distance(p1, p3) < tolerance or distance(p2, p3) + distance(p3, p1) - distance(p2, p1) < tolerance or distance(p3, p1) + distance(p1, p2) - distance(p3, p2) < tolerance


def sign(x):
    if x < 0:
        return -1
    return 1


def distance(p1, p2):
    return math.sqrt(pow(p1[1] - p2[1], 2) + pow(p1[0] - p2[0], 2))


def connect(p1, p2):
    points = list()

    if p1 == p2:
        points.append(p1)
        return points

    if abs(p2[1] - p1[1]) == 0:
        for y in range(p1[0], p2[0]+1 if p1[0] < p2[0]+1 else p2[0]-1, 1 if p1[0] < p2[0]+1 else -1):
            points.append((y, p1[1]))
        return points

    if abs(p2[0] - p1[0]) == 0:
        for x in range(p1[1], p2[1]+1 if p1[1] < p2[1]+1 else p2[1]-1, 1 if p1[1] < p2[1]+1 else -1):
            points.append((p1[0], x))
        return points

    slope = (p2[0] - p1[0]) / (p2[1] - p1[1])
    p2 = (p2[0] - p1[0], p2[1] - p1[1])
    def key(x): return x
    def reverse_key(x): return key(x)
    if slope > 1 and p2[0] > 0:
        def key(x): return (x[1], x[0])
    elif slope < -1 and p2[0] > 0:
        def key(x): return (-x[1], x[0])
        def reverse_key(x): return (x[1], -x[0])
    elif slope < 0 and p2[0] > 0:
        def key(x): return (x[0], -x[1])
    elif slope <= 1 and slope > 0 and p2[0] < 0:
        def key(x): return (-x[0], -x[1])
    elif slope > 1 and p2[0] < 0:
        def key(x): return (-x[1], -x[0])
    elif slope < -1 and p2[0] < 0:
        def key(x): return (x[1], -x[0])
        def reverse_key(x): return (-x[1], x[0])
    elif slope < 0 and p2[0] < 0:
        def key(x): return (x[0], x[1])
    else:
        def key(x): return x

    p2 = key(p2)
    p2 = (p2[0] + p1[0], p2[1] + p1[1])

    deltax = p2[1] - p1[1]
    deltay = p2[0] - p1[0]
    deltaerr = abs(deltay / deltax)
    error = 0
    y = p1[0]
    for x in range(p1[1], p2[1]+1):
        points.append((y, x))
        error = error + deltaerr
        if error >= 0.5:
            points.append((y, x+1))
            y += 1 if p1[0] < p2[0] else -1
            error = error - 1.0
    points = [(x[0] - p1[0], x[1] - p1[1]) for x in points]
    points = [(reverse_key(x)[0] + p1[0], reverse_key(x)[1] + p1[1])
              for x in points]
    return points


def tangent(center, p):
    slope = -(p[1] - center[1]) / (p[0] - center[0])
    return abs(math.tan(slope))


def find_corners(contour, min_angle_perc, max_angle, chord_lengths, image):
    maxs = [0] * len(chord_lengths)
    angle_sets = list()
    for p in range(len(contour)):
        angles = list()
        for C in range(len(chord_lengths)):
            l = chord_lengths[C]
            total_d = 0
            for k in range(p - l, p):
                chord = (contour[len(contour)-1+k] if k < 0 else contour[k],
                         contour[(k+l)-len(contour)-1] if k+l > len(contour)-1 else contour[k + l])
                if chord[1][1] - chord[0][1] == 0:
                    d = abs(chord[1][1] - contour[p][1])
                else:
                    slope = (chord[1][0] - chord[0][0]) / \
                        (chord[1][1] - chord[0][1])
                    a = slope
                    b = -1
                    c = chord[0][0] + slope * -chord[0][1]
                    d = (abs(a * contour[p][1] + b * contour[p]
                             [0] + c)) / (math.sqrt(pow(a, 2) + pow(b, 2)))
                total_d += d
            angles.append(total_d)
            if total_d > maxs[C]:
                maxs[C] = total_d
        angle_sets.append((angles, contour[p][0], contour[p][1]))

    corners = list()
    corner_group = list()
    best_corner = None
    greatest_angle = 0
    index = 0
    for ang_set in angle_sets:
        angle = 1
        for a in range(len(ang_set[0])):
            if maxs[a] == 0:
                angle = 0
            else:
                angle *= ang_set[0][a] / maxs[a]
        if angle > min_angle_perc:
            # image[ang_set[1], ang_set[2]] = (0, 255, 0)
            if index == len(angle_sets)-1 or (len(corner_group) > 0 and distance(ang_set[1:], corner_group[len(corner_group) - 1][1:]) > 2):
                corner_group.clear()
                corners.append(best_corner)
                greatest_angle = 0
                image[best_corner[1], best_corner[2]] = (0, 255, 0)
            corner_group.append(ang_set)
            if angle > greatest_angle:
                greatest_angle = angle
                best_corner = ang_set
        if index == len(angle_sets) - 1:
            corners.append(best_corner)
            image[best_corner[1], best_corner[2]] = (0, 255, 0)
        index += 1

    final_corners = list()
    for a in range(len(corners)):
        if a == 0:
            d = corners[-1][1:]
        else:
            d = corners[a - 1][1:]
        c = corners[a][1:]
        if a == len(corners) - 1:
            e = corners[0][1:]
        else:
            e = corners[a + 1][1:]

        if a == 0:
            ind = int((contour.index(d) +
                       (len(contour) - 1 + contour.index(c))) / 2)
            m1 = contour[ind - len(contour) -
                         1] if ind > len(contour) - 1 else contour[ind]
            # image[m1[0], m1[1]] = (0, 255, 0)
            # image[c[0], c[1]] = (255, 0, 0)
        else:
            m1 = contour[int((contour.index(d) + contour.index(c)) / 2)]
        if collinear(d, m1, c):
            left_tan = get_angle(c, d)
        else:
            c1 = fit_circle(d, m1, c, np.copy(image))
            B1 = get_angle(c, c1)
            P1 = get_angle(c, m1)
            left_tan = B1 + sign(math.sin(P1 - B1)) * (math.pi / 2)

        if a == len(corners)-1:
            ind = int((contour.index(c) +
                       (len(contour) - 1 + contour.index(e))) / 2)
            m2 = contour[ind - len(contour) -
                         1] if ind > len(contour) - 1 else contour[ind]
            # image[c[0], c[1]] = (255, 0, 0)
            # image[m2[0], m2[1]] = (0, 255, 0)
        else:
            m2 = contour[int((contour.index(e) + contour.index(c)) / 2)]
        if collinear(e, m2, c):
            right_tan = get_angle(c, e)
        else:
            # print("non colin")
            c2 = fit_circle(e, m2, c, np.copy(image))
            B2 = get_angle(c, c2)
            P2 = get_angle(c, m2)
            right_tan = B2 + sign(math.sin(P2 - B2)) * (math.pi / 2)

        angle = abs(left_tan-right_tan) if abs(left_tan -
                                               right_tan) < math.pi else math.pi * 2 - abs(left_tan - right_tan)

        if math.degrees(angle) <= max_angle:
            # image[c[0], c[1]] = (255, 0, 0)
            convex = False

            right_tan = right_tan if right_tan >= 0 else right_tan + math.pi * 2
            left_tan = left_tan if left_tan >= 0 else left_tan + math.pi * 2
            right_tan %= math.pi*2
            left_tan %= math.pi * 2
            mid1 = (right_tan + left_tan) / 2
            mid2 = mid1 - math.pi if mid1 > math.pi else mid1 + math.pi
            if right_tan > left_tan:
                image[c[0], c[1]] = (255, 0, 0)
                convex = True
            else:
                image[c[0], c[1]] = (0, 0, 255)

            if min(abs(left_tan - mid1), abs(right_tan - mid1)) > min(abs(left_tan - mid2), abs(right_tan - mid2)):
                mid = mid2
            else:
                mid = mid1

            # print(str(math.degrees(mid1)) + " " + str(mid2) + " " + str(mid))

            vector = (round(math.sin(mid)),
                      round(math.cos(mid)))

            # if image[c[0] + vector[0], c[1] + vector[1]] > 100:
            final_corners.append((c, angle, vector, convex))

            # line = connect(c, (c[0] + vector[0] * 5, c[1] + vector[1] * 5))
            # for p in line:
            # image[p[0], p[1]] = 150

            # image[m1[0], m1[1]] = (0, 255, 0)
            # print(str(c[0] + y2) + " " + str(c[1] + x2))
            # line = connect(c, (c[0] + vector[0] * 5, c[1] + vector[1] * 5))
            # print(len(line))
            # for p in line:
            # image[p[0], p[1]] = 150

            x1 = round(5 * math.cos(right_tan))
            y1 = round(5 * math.sin(right_tan))
            # image[m2[0], m2[1]] = (0, 255, 0)
            # print(str(c[0] + y1) + " " + str(c[1] + x1))
            line = connect(c, (c[0] + y1, c[1] + x1))
            # print(len(line))
            # for p in line:
            # image[p[0], p[1]] = (255, 0, 0)

            x1 = round(5 * math.cos(left_tan))
            y1 = round(5 * math.sin(left_tan))
            # image[m2[0], m2[1]] = (0, 255, 0)
            # print(str(c[0] + y1) + " " + str(c[1] + x1))
            line = connect(c, (c[0] + y1, c[1] + x1))
            # print(len(line))
            # for p in line:
            # image[p[0], p[1]] = (0, 0, 255)

    return final_corners


def distance_from_line(point, line):
    slope = (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])
    b = -slope * line[0][0] + line[0][1]
    return abs((slope * point[0] + b) - point[1])


def straighten(image):
    img = image.copy().astype(np.uint8)
    binary = cv2.Canny(img, 100, 200)
    # threshold = get_threshold(image)*1.3
    # ret, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    # Image.fromarray(binary.astype("uint8"), "L").show()
    minLineLength = int(math.sqrt(image.shape[1]))
    maxLineGap = 5
    # print(int(math.sqrt(image.shape[1])*5))
    lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180, threshold=int(math.sqrt(image.shape[1])*5),
                            minLineLength=minLineLength, maxLineGap=maxLineGap)

    try:
        if lines == None:
            color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            line_height = int(image.shape[0]-image.shape[0]/10)
            color_image[line_height, :] = (0, 0, 255)
            #Image.fromarray(color_image.astype("uint8"), "RGB").show()
            return image, line_height
    except:
        pass

    # print(len(lines))

    img = image.copy()
    # print(len(lines))
    line_groups = list()
    for l in lines:
        x1, y1, x2, y2 = l[0]
        close_lgs = list()
        # Image.fromarray(img.astype("uint8"), "L").show()
        # img = image.copy()
        for lg in line_groups:
            slope = (lg[1][1] - lg[0][1]) / (lg[1][0] - lg[0][0]) / 2
            b = -slope * lg[0][0] + lg[0][1]
            dist = abs((slope * x1 + b) - y1) + abs((slope * x2 + b) - y2)
            # cv2.line(img, (x1, y1),
            #         (x1, int(slope * x1 + b)), 0, 2)
            # cv2.line(img, (x2, y2),
            #         (x2, int(slope * x2 + b)), 0, 2)
            cv2.line(img, (x1, y1), (x2, y2), 0, 2)
            # cv2.line(img, lg[0], lg[1], 100, 2)
            # Image.fromarray(img.astype("uint8"), "L").show()
            # time.sleep(1)
            # img = image.copy()
            # print(dist)
            if dist < 6:
                close_lgs.append(lg)
        # if current_lg != None:
            # print(line_groups.index(current_lg))
        if len(close_lgs) == 0:
            # print("new")
            line_groups.append([(x1, y1), (x2, y2), 0])
            # cv2.line(img, (x1, y1), (x2, y2), 100, 2)
        else:
            for lg in close_lgs:
                if x1 < lg[0][0]:
                    lg[0] = (x1, y1)
                if x2 > lg[1][0]:
                    lg[1] = (x2, y2)
            lg[2] += 1
    # print(len(line_groups))
    #Image.fromarray(img.astype("uint8"), "L").show()
    img = image.copy()
    best_lg = line_groups[0]
    best_dist = 0
    for lg in line_groups:
        cv2.line(img, lg[0], lg[1], 100, 2)
        #Image.fromarray(img.astype("uint8"), "L").show()
        img = image.copy()
        dist = abs(lg[0][0] - lg[1][0]) + lg[2]
        if dist > best_dist:
            best_lg = lg
            best_dist = dist
    e1 = best_lg[0]
    e2 = best_lg[1]
    cv2.line(img, e1, e2, 0, 2)
    #Image.fromarray(img.astype("uint8"), "L").show()

    slope = (e2[1]-e1[1])/(e2[0]-e1[0])
    rot = math.degrees(math.atan(slope))
    image_rotated = Image.fromarray(image.astype("uint8"), "L")
    image_rotated = image_rotated.rotate(rot, fillcolor="white")
    # image_rotated.show()
    line_height = int((e1[1] + e2[1]) / 2)
    color_image = cv2.cvtColor(np.array(image_rotated), cv2.COLOR_GRAY2RGB)
    color_image[line_height, :] = (0, 0, 255)
    #Image.fromarray(color_image.astype("uint8"), "RGB").show()
    return np.array(image_rotated), line_height


def remove_lines(image):
    kernel = np.ones((1, min(image.shape[1], int(2000/20))), np.uint8)
    lines = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # Image.fromarray(lines.astype("uint8"), "L").show()
    return np.array(cv2.add(image, (255 - lines)))


def run_on_line(form, line, pencil=False):
    #global canvas
    image = Image.open(
        "CustomHandwritingDataset/Lines/form{}/form{}-line{}.jpg".format(form, form, line)).convert("L")
    # setting blur = True will have detremental effects on pen handwriting because it closes small holes, only use for pencil
    strokes = extract_strokes(image, pencil)


def extract_strokes(image, blur=False, cursive=False):
    image = image.convert("L")
    width = (image.size[0]/3000) * 2000
    image.thumbnail((width, 300))
    # image.show()
    # print(image.size)
    image = np.array(image)
    image, line_height = straighten(image)
    if blur:
        image = cv2.medianBlur(image, 5)
    image = remove_lines(image)
    thresh = get_threshold(image)*0.975
    threshold_inds = image < thresh
    image = np.full(image.shape, 255)
    image[threshold_inds] = 0
    image = np.array(image, np.uint8)
    image = np.pad(image, 1, "constant", constant_values=255)
    image = convert4connected(image)
    image = remove_holes(image)
    if not cursive:
        image = seperate_letters(image)
    strokes = thin(image, line_height)
    return strokes


def draw_line(x1, y1, x2, y2):
    canvas.create_line(x1, y1, x2, y2)
    canvas.update()


if __name__ == "__main__":
    # can be used to visualize the lines as they are traced
    """window = tk.Tk()
    canvas = tk.Canvas(window, width=1000, height=300)
    canvas.pack()
    window.after(2000, run_on_line(72, 12, True))
    window.mainloop()"""

    run_on_line(1, 4, False)

