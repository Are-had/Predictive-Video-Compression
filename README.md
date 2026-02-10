# Predictive Video Compression Engine

A Python implementation of a **Motion Compensated Prediction** pipeline built from scratch.

This project demonstrates the core principles of modern video coding (like MPEG/H.264). It implements a **Block Matching Algorithm (BMA)** to exploit temporal redundancy between frames, effectively compressing video data by storing motion vectors and residual errors instead of full raw frames.

## Project Overview

The goal is to simulate the internal logic of a video codec to understand the trade-offs between **compression ratio** and **reconstruction quality**.

The engine performs three main steps for every frame:
1.  **Motion Estimation:** Finds where blocks of pixels have moved compared to the previous frame (using Full Search).
2.  **Motion Compensation:** Reconstructs the current frame using only the reference frame and the calculated motion vectors.
3.  **Residual Calculation:** Computes the difference (error) between the prediction and reality to ensure quality.

##  Key Results

Running the algorithm on the test sequence yielded the following performance metrics:

* **Compression Ratio:** **3.32x** (Raw pixel data vs. Compressed `.npz` stream)
* **Reconstruction Quality:** **~23.65 dB** (Average PSNR)
* **Method:** Full Search Block Matching (16x16 macroblocks)

## Configuration

You can tweak the compression parameters in `main.py`:

```python
BLOCK_SIZE = 16      # Size of the macroblocks (e.g., 8 or 16)
SEARCH_AREA = 7      # How far to search for matching blocks (larger = slower but better)
RESIZE_WIDTH = 320   # Downscaling width for performance