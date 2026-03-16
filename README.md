# OIC-274 Quantifying DNA damage

The goal of this project is to quantify DNA damage in cells. DNA damage is visible as puncta formed by a phospho-histone marker coupled to an Alexa fluorophore.

## Getting started

These instructions assume that the image are collected as CZI-files (Zeiss).

### Prerequisites

- [Python](https://www.python.org/downloads/) version 3.13.7

### Download code

1. Download or clone the GitHub repository
   ```bash
   git clone git@github.com:vaioic/OIC-274.git
   cd OIC-274
   ```

### Python setup

If running the code for the first time, you will need to create a Python virtual environment and install the necessary packages. 

1. Open a terminal and navigate to the directory where you unzipped the files.

2. Create a python virtual environment
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment
   Windows:
   ```bash
   .\venv\Scripts\activate
   ```
   
   Linux:
   ```bash
   source venv/scripts/activate
   ```

4. Install the dependencies using Pip
   ```bash
   python -m pip install -r .\requirements.txt
   ```

### Running the code

1. Start the virtual environment if not already loaded
   ```bash
   .\venv\Scripts\activate
   ```

2. Call ``process_images()`` to process images. In the current version, the directory to the images is hardcoded so you will need to edit the lines:
   ```python
   if __name__ == "__main__":
    process_files_in_dir(r"D:\Projects\OIC-274 Rahma\data\03042026", r"D:\Projects\OIC-274 Rahma\processed\2026-03-16")
   ```

   The function ``process_files_in_dir`` takes two inputs: The first is the directory to the images, and the second is the output directory. **This functionality will be removed with a normal input parser at a later date.**

### Results

The code will output the following files:

1. A overlay image showing each cell, label, and identified spot in the image. An image file is created for each input image.
2. ``results.csv`` is a CSV file containing the following columns:
   - image: Name of image file
   - cell_label: Label of the cell in an image
   - spot_label: Label of the spot in an image
   - intensity_mean: Mean intensity of each spot (in arb. units)
   - spot_diameter: Diameter of the spot (in pixels)
3. ``summary.csv`` is a CSV-file containing the following columns:
   - image: Name of image file
   - cell_label: Label of the cell in an image
   - mean_intensity: The average of mean intensities of all spots in a cell
   - num_spots: Number of spots in a cell
4. ``results.nc`` is a netCDF file containing an xarray holding the data
   - coords:
         - image
         - cell_label
         - spot_label
   - data:
         - intensity_mean
         - max_feret_diameter

## Issues

If you encounter any issues with running the code or have any questions, please create an [Issue](https://github.com/vaioic/OIC-274/issues) or send an email to opticalimaging@vai.org. If you are reporting a programmatic bug, please include any error messages to aid with troubleshooting.

## Acknowledgements

### Contributors
<a href="https://github.com/vaioic/OIC-262/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=vaioic/OIC-262" />
</a>

### Dependencies

This project relies primarily on the following packages:

* xarray v2026.2.0
* scikit-image v0.26.0

**Note:** For full dependency list, see [requirements.txt](requirements.txt).