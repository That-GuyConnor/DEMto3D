# DSM2STL: DSM to 3D Printable Model Converter

DSM2STL is a Python-based tool that converts Digital Surface Models (DSM) in GeoTIFF format into watertight, 3D printable STL models. It is specifically designed for working with geospatial data (e.g., NAD83 in meters) and includes advanced preprocessing steps—such as flood fill for ocean detection and median filtering for outlier removal—to generate accurate and aesthetically pleasing 3D models.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
  - [DEM Preprocessing](#dem-preprocessing)
  - [Mesh Generation](#mesh-generation)
  - [Resolution Control](#resolution-control)
  - [Progress Monitoring & Logging](#progress-monitoring--logging)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

DSM2STL converts a DSM stored in a GeoTIFF file (e.g., `SF.tif`) into a 3D printable STL model. The tool:
- Removes ocean areas from the DSM using a flood-fill algorithm.
- Eliminates outlier elevation cells with a median filter.
- Creates a watertight mesh by generating surface faces only for land cells, extruding walls along the land boundary, and adding a base cap.
- Adjusts the mesh resolution according to a specified nozzle size, ensuring that unnecessary geometry is not generated.
- Supports a test mode that downsamples the DSM to a fixed resolution for rapid prototyping.

---

## Features

- **Ocean Detection:** Uses flood-fill on the DEM edges to detect and mask ocean areas.
- **Outlier Removal:** Applies median filtering to remove spurious peaks and smooth the data.
- **Nozzle-Based Resolution:** Computes mesh grid resolution based on your printer's nozzle size (in mm) to avoid over-detailing.
- **Test Mode:** Enables quick iterations by downsampling the grid to a fixed 500×500 resolution.
- **Watertight Mesh Generation:** Creates a surface mesh, extrudes vertical walls along the land boundary, and triangulates a base cap.
- **Extensive Logging & Progress Bars:** Detailed logging at various levels (DEBUG, INFO, etc.) and progress bars throughout the mesh creation process.
- **Tile Support:** Split the DSM into tiles for easier handling and printing of large areas.

---

## Requirements

- Python 3.6+
- Required Python packages:
  - `numpy`
  - `rasterio`
  - `trimesh`
  - `tqdm`
  - `scipy`
  - `scikit-image`
  - `shapely`

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/DSM2STL.git
   cd DSM2STL
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install the Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, install the packages manually:

   ```bash
   pip install numpy rasterio trimesh tqdm scipy scikit-image shapely
   ```

---

## Usage

1. **Place Your DSM GeoTIFF:**  
   Ensure your DSM file (e.g., `SF.tif`) is located in the repository directory or update the `INPUT_FILE` parameter in the script.

2. **Configure Settings:**  
   Adjust parameters in the script (e.g., target width, nozzle size, test mode, etc.) to suit your specific needs.

3. **Run the Script:**

   ```bash
   python dsm2stl.py
   ```

4. **Output:**  
   The generated STL files will be saved in the designated output directory (default: `output_tiles`).

---

## Configuration

All configuration options are set directly within the script. Key parameters include:

- **Input/Output:**
  - `INPUT_FILE`: Path to the DSM GeoTIFF (e.g., `SF.tif`).
  - `OUTPUT_DIR`: Directory where STL files will be saved.

- **Model Dimensions:**
  - `TARGET_WIDTH`: Desired printed width (in mm) of each tile.
  - `Z_SCALE`: Vertical exaggeration factor (default is `1.0`).
  - `BASE_THICKNESS`: Thickness (in mm) of the base below the land surface.

- **Tiling:**
  - `TILE_ROWS` and `TILE_COLS`: Number of tiles to split the DSM into.

- **Resolution Control:**
  - `NOZZLE_SIZE`: Sets the resolution based on the nozzle diameter (in mm).
  - `test_mode`: When `True`, the DSM is downsampled to a fixed 500×500 grid for faster processing.

- **Preprocessing:**
  - `WATER_TOLERANCE`: Tolerance (in meters) used during flood fill to detect ocean cells.
  - `MEDIAN_FILTER_SIZE`: Kernel size for the median filter to remove outlier cells.

- **Logging:**
  - `LOG_LEVEL`: Set to `logging.DEBUG` for detailed logs or adjust as needed.

---

## How It Works

### DEM Preprocessing

1. **Flood Fill for Ocean Detection:**  
   The script uses a flood-fill algorithm starting at the DEM edges to mark cells with low elevation (within the specified `WATER_TOLERANCE`) as ocean. These cells are then masked out, ensuring that the ocean does not contribute to the final mesh.

2. **Outlier Removal:**  
   A median filter is applied to the DEM data to smooth out spurious elevation peaks that can lead to unwanted geometry in the final model.

### Mesh Generation

1. **Grid Construction:**  
   Based on the physical extents derived from the DSM (using pixel size from the GeoTIFF) and the specified `TARGET_WIDTH`, the tool calculates a conversion factor (mm per meter).  
   The DSM is then resampled to a grid resolution determined either by fixed divisions (in test mode) or computed based on the `NOZZLE_SIZE`.

2. **Surface Faces:**  
   The script generates surface faces (triangles) only for cells where all four vertices are classified as land. This ensures that only valid terrain is included in the final model.

3. **Boundary Extraction and Wall/Base Generation:**  
   - **Land Boundary:** The boundary of the land area is extracted using the `find_contours` function.
   - **Wall Generation:** Vertical walls are extruded along the land boundary, creating a transition from the terrain to a flat base.
   - **Base Cap:** The boundary is triangulated to form a base cap, ensuring that the mesh is watertight.

### Resolution Control

- **Nozzle-Based Resolution:**  
  When not in test mode, the final grid resolution is computed based on the nozzle size (NOZZLE_SIZE). This prevents the generation of excessive geometry by ensuring that the mesh detail is appropriate for the 3D printer’s capabilities.

- **Test Mode:**  
  A test mode is provided to force the DSM to a fixed 500×500 resolution for quick iterations and debugging.

### Progress Monitoring & Logging

- **Progress Bars:**  
  The script uses `tqdm` to display progress bars during surface face generation, wall face creation, and base cap triangulation.

- **Logging:**  
  Extensive logging (using Python’s `logging` module) provides detailed diagnostic information at each step, from data loading and preprocessing to mesh generation and export.

---

## Example

Assuming you have a DSM file named `SF.tif` in your project directory, simply run:

```bash
python dsm2stl.py
```

You will see progress bars in the console indicating the progress of surface face generation, wall extrusion, and base cap triangulation. Detailed logs will be output, and the final STL model(s) will be located in the `output_tiles` directory.

---

## Contributing

Contributions to DSM2STL are welcome! If you have ideas for improvements, bug fixes, or additional features, please:

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Submit a pull request with a detailed description of your changes.

Please adhere to the coding style used in the project and include appropriate tests and documentation.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
