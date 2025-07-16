# Online Handwriting Dataset Generation Pipeline

![linesegmentation_cropped](https://github.com/user-attachments/assets/f627ba7e-3d0d-4c8f-9448-d78631689b88)

## Project Overview

This project provides a complete pipeline to convert offline handwritten forms (scanned bitmap images) into online handwriting data—sequences of (x, y, pen-state) points. The pipeline includes automated text-line and word segmentation, skeleton-based stroke extraction, and dataset formatting for machine learning applications. 

The generated dataset can be used for handwriting analysis, recognition, or generative modeling. I used this dataset to train a handwriting synthesis model using [sjvasquez/handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis), based on Alex Graves' paper [*Generating Sequences with Recurrent Neural Networks*](https://arxiv.org/abs/1308.0850).

> See my [CNC Pen Plotter project](https://github.com/Carson-Stark/CNCPenPlotter), which was intended to use handwriting synthesis for rendering realistic handwritten notes on physical paper in my style.

### Features

- Automated text-line and word segmentation
- Skeleton-based stroke extraction
- Dataset formatting for machine learning applications

### Project Timeline

- **Date Started**: August 2019  
- **Date Completed**: May 2020

## Example Results

### Word / Line Partition

<img width="1920" height="1080" alt="word_segmentation" src="https://github.com/user-attachments/assets/81898c28-5ba5-40db-9198-5f5bce7be594" />

### Stroke Extraction

![stroke_extraction](https://github.com/user-attachments/assets/d3cdd34f-2611-4b4e-b76d-57048e7881fc)

### Handwriting Synthesis Training

![Comparison](https://github.com/user-attachments/assets/4366e724-1947-4369-bf89-a962a4ec1507)

## Installation

To get started, clone the repository:

```
git clone https://github.com/Carson-Stark/HandwrittingSynthesis.git
cd HandwrittingSynthesis
```

Ensure you have Python 3.x installed. Then install the required packages with:

```bash
pip install numpy Pillow opencv-python
```

## Usage

Prepare your dataset by placing scanned handwritten forms in the `CustomHandwritingDataset/` folder structured as:

```
Root Directory
├── CustomHandwritingDataset/
│   ├── Forms/ # Scanned form images (input)
│   ├── Lines/ # Output line images (generated)
│   ├── Strokes/ # Output stroke data (generated)
│   ├── Transcriptions/ # Text transcriptions per form
```

> **Note**: `CustomHandwritingDataset/` contains sample data for a single form. The rest of the dataset is excluded for privacy reasons. You must supply your own data.

### Required Input Data:

- **Forms/**:  
  High-resolution scanned images of handwritten pages. They can be in formats like PNG or JPEG, color or grayscale.  
  Each image should represent a single handwritten page.

- **Transcriptions/**:  
  Text files corresponding to each form image. Each `.txt` file should have the same filename (excluding extension) as the form image.  
  Each line in the `.txt` file should match one handwritten line in the form image, in top-to-bottom order. Recommended to generate line partitions first, then manually adjust the text files to match the segmented lines. Feel free to experiment with OCR tools to automate this step, but manual verification is recommended for accuracy.

### Workflow

1. **Gather Forms**  
   Add images of individual pages of your handwriting into `Forms/`. Must be JPG format, cropped closely to page, and straight on perspective. No further processing required.

3. **Rename Forms**  
   Use `data_formater.py` to rename forms as `form1.jpg`, `form2.jpg`, etc.

4. **Segment Lines**  
   Use `data_formater.py` segmentLines() function to perform automatic line segmentation on the forms and save into `CustomHandwritingDataset/Lines/`. `textSegmentation.py` can also segment into words depending on configuration.

5. **Extract Strokes**  
   Use `data_formater.py` extractStrokes() function to convert each line image into pen-stroke sequences and save into `CustomHandwritingDataset/Strokes/` Use `test_points.py` to preview the extracted (x, y, pen-state) sequences.

6. **Train Model**  
   Train a handwriting synthesis or other model using this dataset. I used [this repo](https://github.com/sjvasquez/handwriting-synthesis) to get the results above. Ensure you convert the strokes data into the expected format if needed.

## Project Structure

```
Root Directory
├── .gitignore # Ignores CustomHandwritingDataset/
├── README.md # Project documentation (this file)
├── data_formater.py # Data formatting script
├── textSegmentation.py # Line and word segmentation script
├── skeletionize.py # Stroke extraction script
├── test_points.py # Visualization script
└── CustomHandwritingDataset/ # User-provided data (not included)
```

## Algorithm Overview

### 1. Text-Line and Word Segmentation

The segmentation process is based on the method described in *A New Scheme for Unconstrained Handwritten Text-Line Segmentation* by Alaei, Pal, and Nagabhushan.  
[View Paper](http://researchgate.net/publication/220599927_A_new_scheme_for_unconstrained_handwritten_text-line_segmentation)

#### Core steps:

- **Vertical Strip Decomposition**  
  The page is divided into vertical strips to localize text structure at finer granularity.

- **Piece-wise Painting**  
  Each strip is "painted" by smoothing row pixel densities to emphasize separations between lines.

- **Dilation**  
  Apply dilation to merge connected components within each text line and reduce fragmentation.

- **Background Thinning**  
  Thin the background regions between text to produce candidate line separators.

- **Separator Linking**  
  Connect line separators across strips to form continuous curves that divide lines.

- **Overlap Handling**  
  In areas where lines touch or overlap, local separator curves are inserted to correct merging errors.

- **Final Line and Word Extraction**  
  Use the separator paths to extract individual line images, then apply connected components and spacing heuristics to segment words.

### 2. Pen-Stroke Extraction (Skeletonization)

Stroke extraction is based on *Recovery of Drawing Order From Single-Stroke Handwriting Images* by Kimura et al.  
[IEEE Link](https://ieeexplore.ieee.org/document/877517)

#### Core steps:

- **Skeletonization**  
  Apply morphological thinning to reduce the handwriting to a 1-pixel-wide skeleton.

- **Graph Construction**  
  Model the skeleton as a graph where nodes represent junctions and endpoints, and edges represent line segments.

- **Stroke Path Reconstruction**  
  Traverse the graph to produce ordered stroke sequences approximating pen motion, with inferred pen-up/pen-down states.

- **Sequence Formatting**  
  Output each stroke as a sequence of (x, y, pen-state) tuples ready for input to handwriting models.
