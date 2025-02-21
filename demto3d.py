#!/usr/bin/env python3
import os
import numpy as np
import rasterio
import logging
from tqdm import tqdm
import trimesh
from scipy.ndimage import zoom, median_filter
from skimage.measure import find_contours
from shapely.geometry import Polygon
from trimesh.creation import triangulate_polygon
from scipy.interpolate import RectBivariateSpline
from collections import deque

# =======================
# CONFIGURATION
# =======================
INPUT_FILE = 'SF.tif'         # Input DSM GeoTIFF file (in NAD83, units in meters)
OUTPUT_DIR = 'output_tiles'   # Directory to save output STL tiles
TARGET_WIDTH = 1000.0          # Printed width (in mm) of each tile
Z_SCALE = 1.5                 # Additional vertical exaggeration (1.0 = no exaggeration)
TILE_COLS = 2                 # Number of tiles horizontally
TILE_ROWS = 2                 # Number of tiles vertically
BASE_THICKNESS = 2.0          # How far (in mm) below the lowest land elevation the base will be

# -------------------------------
# TEST MODE / DOWNSAMPLING CONFIGURATION
# -------------------------------
test_mode = False            # Set False to use full resolution based on nozzle size
if test_mode:
    fixed_divisions = (500, 500)  # Force a 500x500 grid in test mode
else:
    fixed_divisions = None

# -------------------------------
# NOZZLE SIZE CONFIGURATION (in mm)
# -------------------------------
NOZZLE_SIZE = 0.2  # Determines the desired resolution (minimum geometry size)

# -------------------------------
# OCEAN DETECTION / OUTLIER REMOVAL CONFIGURATION
# -------------------------------
WATER_TOLERANCE = 0.5  # In meters; used for flood fill to mark ocean cells
MEDIAN_FILTER_SIZE = 3 # Use a 3x3 median filter to smooth out spurious peaks

# -------------------------------
# Logging Levels:
# DEBUG: Detailed diagnostic information.
# INFO: Confirmation that things are working as expected.
# WARNING: Something unexpected happened.
# ERROR: A serious problem prevented some functionality.
# CRITICAL: A severe error indicating the program may not continue.
# -------------------------------
LOG_LEVEL = logging.DEBUG

# =======================
# SETUP LOGGING
# =======================
def setup_logging():
    logging.basicConfig(level=LOG_LEVEL,
                        format='%(asctime)s - %(levelname)s - %(message)s')

# =======================
# DEM PRE-PROCESSING: Detect Ocean via Flood Fill
# =======================
def detect_ocean(dem, tolerance=WATER_TOLERANCE):
    """
    Perform a flood fill from the DEM’s borders to identify ocean cells.
    Any cell connected to a border cell with an elevation below (min_border + tolerance)
    is marked as ocean. Returns a boolean mask (True = land, False = ocean).
    """
    nrows, ncols = dem.shape
    ocean = np.zeros((nrows, ncols), dtype=bool)
    q = deque()
    # Determine water level from the border cells (using the minimum)
    border_vals = np.concatenate([dem[0, :], dem[-1, :], dem[:, 0], dem[:, -1]])
    water_level = np.min(border_vals)
    # Enqueue border cells that are near water level
    for i in range(nrows):
        for j in [0, ncols-1]:
            if dem[i, j] <= water_level + tolerance and not ocean[i, j]:
                ocean[i, j] = True
                q.append((i, j))
    for j in range(ncols):
        for i in [0, nrows-1]:
            if dem[i, j] <= water_level + tolerance and not ocean[i, j]:
                ocean[i, j] = True
                q.append((i, j))
    # Flood fill
    while q:
        i, j = q.popleft()
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < nrows and 0 <= nj < ncols and not ocean[ni, nj]:
                if dem[ni, nj] <= water_level + tolerance:
                    ocean[ni, nj] = True
                    q.append((ni, nj))
    # Land mask is the inverse of ocean
    land = ~ocean
    return land

# =======================
# REMOVE OUTLIERS via Median Filtering
# =======================
def remove_outliers(dem, size=MEDIAN_FILTER_SIZE):
    """
    Apply a median filter to the DEM to smooth out isolated spurious peaks.
    """
    filtered = median_filter(dem, size=size)
    return filtered

# =======================
# GENERATE MESH FOR A TILE (LAND-ONLY)
# =======================
def generate_tile_mesh(tile_data, target_width, z_scale, base_thickness, pixel_size_x, pixel_size_y, original_shape=None):
    """
    Generate a watertight 3D mesh for a tile by:
      • Removing ocean cells (using flood fill)
      • Smoothing out outliers via a median filter
      • Generating a grid of vertices (with resolution based on nozzle size or fixed divisions)
      • Creating faces only for grid cells that are entirely land
      • Extracting the boundary of the land area to extrude vertical walls and triangulate a base cap.
    
    Parameters:
      tile_data     - 2D numpy array of elevations (in meters) for this tile.
      target_width  - Printed tile width in mm.
      z_scale       - Vertical exaggeration factor.
      base_thickness- Base thickness (in mm) below the lowest land elevation.
      pixel_size_x  - Real-world pixel width (m).
      pixel_size_y  - Real-world pixel height (m).
      original_shape- Original shape of the tile (before any resampling).  
    """
    # Use original shape for physical extents if provided
    if original_shape is None:
        original_shape = tile_data.shape
    orig_nrows, orig_ncols = original_shape

    # Compute real-world dimensions of the tile (in meters)
    real_width_m = orig_ncols * pixel_size_x
    real_height_m = orig_nrows * pixel_size_y

    # Conversion factor: how many mm per meter so that printed width is TARGET_WIDTH
    conversion_factor = target_width / real_width_m
    target_length = real_height_m * conversion_factor

    # ---------------------------
    # Preprocess DEM: Remove outliers and detect ocean
    # ---------------------------
    tile_data = remove_outliers(tile_data)
    land_mask = detect_ocean(tile_data)  # True = land, False = ocean

    # ---------------------------
    # Resample tile data and land mask to desired grid resolution
    # ---------------------------
    if fixed_divisions is not None:
        divisions_x, divisions_y = fixed_divisions
        logging.debug("Test mode: Resampling tile from shape %s to fixed divisions (%d, %d)",
                      tile_data.shape, divisions_x, divisions_y)
    else:
        divisions_x = int(np.ceil(target_width / NOZZLE_SIZE))
        divisions_y = int(np.ceil(target_length / NOZZLE_SIZE))
        logging.debug("Nozzle-based resolution: Resampling tile from shape %s to (%d, %d) divisions based on nozzle size %.2f mm",
                      tile_data.shape, divisions_x, divisions_y, NOZZLE_SIZE)
    zoom_factor_y = divisions_y / tile_data.shape[0]
    zoom_factor_x = divisions_x / tile_data.shape[1]
    # Use spline interpolation for elevation and nearest for mask
    tile_data = zoom(tile_data, (zoom_factor_y, zoom_factor_x), order=3)
    land_mask = zoom(land_mask.astype(float), (zoom_factor_y, zoom_factor_x), order=0) >= 0.5

    nrows, ncols = tile_data.shape

    # ---------------------------
    # Build grid coordinates (in mm)
    # ---------------------------
    xs = np.linspace(0, target_width, num=ncols)
    ys = np.linspace(0, target_length, num=nrows)
    xs, ys = np.meshgrid(xs, ys)
    # Compute z values (elevation in mm)
    zs = tile_data * conversion_factor * z_scale

    # ---------------------------
    # Build surface faces only for land cells.
    # For each cell (quad) defined by vertices at (i,j), (i,j+1), (i+1,j), (i+1,j+1),
    # only add faces if all four corners are land.
    # ---------------------------
    cell_mask = (land_mask[:-1, :-1] & land_mask[:-1, 1:] &
                 land_mask[1:, :-1] & land_mask[1:, 1:])
    total_cells = np.sum(cell_mask)
    faces = []
    pbar_surface = tqdm(total=int(total_cells), desc="Generating land surface faces", leave=False)
    for i in range(nrows - 1):
        for j in range(ncols - 1):
            if land_mask[i, j] and land_mask[i, j+1] and land_mask[i+1, j] and land_mask[i+1, j+1]:
                idx = i * ncols + j
                idx_right = idx + 1
                idx_bottom = idx + ncols
                idx_bottom_right = idx_bottom + 1
                faces.append([idx, idx_right, idx_bottom])
                faces.append([idx_right, idx_bottom_right, idx_bottom])
                pbar_surface.update(1)
    pbar_surface.close()
    faces = np.array(faces)
    logging.debug("Land surface mesh: %d vertices, %d faces", nrows*ncols, len(faces))
    surface_vertices = np.column_stack((xs.flatten(), ys.flatten(), zs.flatten()))
    surface_mesh = trimesh.Trimesh(vertices=surface_vertices, faces=faces, process=False)

    # ---------------------------
    # Determine base level from land (if any land exists)
    # ---------------------------
    if np.any(land_mask):
        min_z = np.min(zs[land_mask])
    else:
        min_z = np.min(zs)
    base_z = min_z - base_thickness

    # ---------------------------
    # Extract land boundary using find_contours (on the land mask)
    # ---------------------------
    contours = find_contours(land_mask.astype(float), 0.5)
    if len(contours) == 0:
        logging.warning("No land contour found; returning surface mesh only.")
        return surface_mesh
    # Choose the longest contour (assumed to be the main land boundary)
    longest_contour = max(contours, key=lambda x: len(x))
    # Convert contour coordinates (row, col) to physical (x, y)
    contour_x = (longest_contour[:, 1] / (ncols - 1)) * target_width
    contour_y = (longest_contour[:, 0] / (nrows - 1)) * target_length
    # Interpolate z along the contour from the zs grid
    interp = RectBivariateSpline(np.arange(nrows), np.arange(ncols), zs)
    contour_z = interp(longest_contour[:, 0], longest_contour[:, 1], grid=False)
    
    # ---------------------------
    # Build wall vertices: top follows the land surface; bottom is at base_z.
    # ---------------------------
    wall_top = np.column_stack((contour_x, contour_y, contour_z))
    wall_bottom = np.column_stack((contour_x, contour_y, np.full_like(contour_x, base_z)))
    n_boundary = len(wall_top)
    # Combine wall vertices: first all top, then all bottom.
    wall_vertices = np.vstack((wall_top, wall_bottom))
    
    # Build wall faces (quad between consecutive points, split into two triangles)
    wall_faces = []
    pbar_wall = tqdm(total=n_boundary, desc="Generating wall faces", leave=False)
    for i in range(n_boundary):
        next_i = (i + 1) % n_boundary
        top1 = i
        top2 = next_i
        bot1 = i + n_boundary
        bot2 = next_i + n_boundary
        wall_faces.append([top1, top2, bot2])
        wall_faces.append([top1, bot2, bot1])
        pbar_wall.update(1)
    pbar_wall.close()
    wall_faces = np.array(wall_faces)

    # ---------------------------
    # Triangulate the base cap from the land boundary (projected to XY at base_z)
    # ---------------------------
    boundary_polygon = Polygon(np.column_stack((contour_x, contour_y)))
    try:
        base_cap_mesh = triangulate_polygon(boundary_polygon)
    except Exception as e:
        logging.error("Error triangulating polygon for base cap: %s", e)
        base_cap_mesh = None
    if base_cap_mesh is not None:
        base_cap_vertices = np.column_stack((base_cap_mesh.vertices, np.full(len(base_cap_mesh.vertices), base_z)))
        base_cap_faces = base_cap_mesh.faces
    else:
        base_cap_vertices = np.empty((0, 3))
        base_cap_faces = np.empty((0, 3), dtype=int)

    # ---------------------------
    # Combine the surface, walls, and base cap into one mesh.
    # ---------------------------
    offset_wall = len(surface_vertices)
    offset_base = offset_wall + len(wall_vertices)
    combined_vertices = np.vstack((surface_vertices, wall_vertices, base_cap_vertices))
    combined_faces = np.vstack((
        faces,
        wall_faces + offset_wall,
        base_cap_faces + offset_base
    ))
    final_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces, process=True)
    logging.info("Final land mesh generated: %d vertices, %d faces", len(combined_vertices), len(combined_faces))
    return final_mesh

# =======================
# LOAD DSM DATA WITH RESOLUTION
# =======================
def load_dsm(file):
    logging.info("Loading DSM from %s", file)
    with rasterio.open(file) as src:
        data = src.read(1)  # read first band
        pixel_size_x = src.transform.a
        pixel_size_y = -src.transform.e
    logging.info("DSM loaded with shape %s and pixel size: (%.3f m, %.3f m)", data.shape, pixel_size_x, pixel_size_y)
    return data, (pixel_size_x, pixel_size_y)

# =======================
# PROCESS TILES
# =======================
def process_tiles(dsm_data, pixel_size, tile_rows, tile_cols, target_width, z_scale, base_thickness, output_dir):
    pixel_size_x, pixel_size_y = pixel_size
    nrows, ncols = dsm_data.shape
    tile_height = nrows // tile_rows
    tile_width = ncols // tile_cols
    logging.info("Processing tiles: each tile size (in pixels): %d x %d", tile_height, tile_width)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info("Created output directory: %s", output_dir)

    for row in tqdm(range(tile_rows), desc="Tile Rows"):
        for col in tqdm(range(tile_cols), desc="Tile Columns", leave=False):
            start_row = row * tile_height
            start_col = col * tile_width
            tile = dsm_data[start_row:start_row + tile_height, start_col:start_col + tile_width]
            original_shape = tile.shape  # full-resolution shape for physical extents
            logging.info("Processing tile at row %d, col %d", row, col)
            mesh = generate_tile_mesh(tile, target_width, z_scale, base_thickness,
                                      pixel_size_x, pixel_size_y, original_shape=original_shape)
            output_file = os.path.join(output_dir, f"tile_{row}_{col}.stl")
            mesh.export(output_file)
            logging.info("Exported tile mesh to %s", output_file)

# =======================
# MAIN
# =======================
def main():
    setup_logging()
    logging.info("Starting DSM to 3D printable model conversion (land-only)")
    dsm_data, pixel_size = load_dsm(INPUT_FILE)
    process_tiles(dsm_data, pixel_size, TILE_ROWS, TILE_COLS, TARGET_WIDTH, Z_SCALE, BASE_THICKNESS, OUTPUT_DIR)
    logging.info("All tiles processed and exported successfully.")

if __name__ == "__main__":
    main()
