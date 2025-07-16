"""
Converts handwitten forms into a set of lines and strokes and saved them in the dataset directory.
"""


import re
import os
import textSegmentation
import skeletionize
from PIL import Image
import sys
import numpy as np

forms_dir = "CustomHandwritingDataset/Forms"
lines_dir = "CustomHandwritingDataset/Lines"
strokes_dir = "CustomHandwritingDataset/Strokes"

def sorted_alphanumeric(data):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def rename_forms():
    named = sorted_alphanumeric(
        [f for f in os.listdir(forms_dir) if "form" in f and f.index("form") == 0])
    unnamed = sorted_alphanumeric(
        [f for f in os.listdir(forms_dir) if "form" not in f or f.index("form") != 0])
    count = 1
    for form in named:
        os.rename(os.path.join(forms_dir, form), os.path.join(
            forms_dir, "form" + str(count) + ".jpg"))
        count += 1
    for form in unnamed:
        if form[-4:] == ".jpg":
            os.rename(os.path.join(forms_dir, form), os.path.join(
                forms_dir, "form" + str(count) + ".jpg"))
            count += 1


def get_lines():
    form_count = 1
    line_count = 0
    error_count = 0
    err_file = open(os.path.join(forms_dir, "ErrorDocs.txt"), "w")
    for form in sorted_alphanumeric(os.listdir(forms_dir)):
        if form[-4:] != ".jpg" and form[-4:] != ".JPG":
            continue
        if form_count <= 97:
            form_count += 1
            continue
        im = Image.open(os.path.join(forms_dir, form))
        try:
            lines = textSegmentation.segmentLines(im, 3)
        except:
            error_count += 1
            err_file.write(form + "\n")
        else:
            if not os.path.exists(os.path.join(lines_dir, "F" + form[1:-4])):
                os.makedirs(os.path.join(lines_dir, "F" + form[1:-4]))
            count = 1
            for line in lines:
                line.save(os.path.join(lines_dir,
                                       form[:-4], str(form[:-4]) + "-line" + str(count) + ".jpg"))
                count += 1
            line_count += count
            print("Form {}/{} - {} Lines Saved - {} Error Documents".format(form_count,
                                                                            len(os.listdir(forms_dir)), line_count, error_count), end="\r")
            form_count += 1
    err_file.close()


def get_strokes():
    form_count = 1
    line_count = 0
    error_count = 0
    pencil_forms = [(53, 97)]
    err_file = open(os.path.join(lines_dir, "ErrorDocs.txt"), "w")
    for form in sorted_alphanumeric(os.listdir(lines_dir)):
        if form[-4:] == ".txt":
            continue
        if form_count <= 97:
            form_count += 1
            continue
        if not os.path.exists(os.path.join(strokes_dir, form)):
            os.makedirs(os.path.join(strokes_dir, form))
        for line in sorted_alphanumeric(os.listdir(os.path.join(lines_dir, form))):
            im = Image.open(os.path.join(lines_dir, form, line))
            try:
                if any((form_count >= r[0] and form_count < r[1]) for r in pencil_forms):
                    strokes = skeletionize.extract_strokes(im, True)
                else:
                    strokes = skeletionize.extract_strokes(im, False)
            except:
                error_count += 1
                err_file.write(line + "\n")
            else:
                np.save(os.path.join(strokes_dir, form,
                                     line[:-4] + "-strokes"), strokes)
                line_count += 1
                print("Form {}/{} - {} Lines Saved - {} Error Documents".format(form_count,
                                                                                len(os.listdir(forms_dir)), line_count, error_count), end="\r")
        form_count += 1
    err_file.close()

# rename_forms()
# get_strokes()
# get_lines()
