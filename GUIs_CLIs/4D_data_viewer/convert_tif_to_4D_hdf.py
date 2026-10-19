import time
from pathlib import Path
import h5py
import numpy as np
import algodif.io.loadersaver as losa


scan = "scan_00004"
folder_path = "/nsls2/data/hex/proposals/2025-2/pass-318635/edxd/processed/" + scan + "/tifs_with_bgr/"
output_folder = "/nsls2/data/hex/proposals/2025-2/pass-318635/edxd/converted/" + scan + "/"

losa.make_folder(output_folder)
t0 = time.time()
list_files = losa.find_file(folder_path + "*.tif*")

num_row = 151  # of scan 2D, we can get it from metadata
num_column = 151  # of scan 2D, we can get it from metadata
num_file = len(list_files)
height, width = losa.load_image(list_files[0]).shape
chunks = (64, 64, 8, 8)  # For compromising between fast writing and reading


def convert_tif_scan_to_hdf(input_path, output_path, nrow, ncol,
                            scan_order="row", key_path="entry/data",
                            data_type="float32", crop=(0, 0, 0, 0),
                            chunks=(64, 64, 8, 8), pattern=None,
                            overwrite=True, **options):
    """
    Convert TIFF images from a 2D scan to a 4D HDF5 dataset.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Folder containing TIFF files. Sorted filenames must follow
        acquisition order. Use zero-padded indices, e.g. img_00001.tif.
    output_path : str or pathlib.Path
        Path to the HDF5 file.
    nrow, ncol : int
        Number of rows and columns in the scan grid.
    scan_order : {'row', 'column', 'snake_row', 'snake_column'}, optional
        Acquisition order used to map TIFF files to scan positions:

        - 'row': Complete each row before moving to the next row.
        - 'column': Complete each column before moving to the next column.
        - 'snake_row': Scan by rows, reversing alternate rows.
        - 'snake_column': Scan by columns, reversing alternate columns.

        The first image is assigned to (row=0, column=0).
        The first row or column follows increasing index order.
    key_path : str, optional
        Key path to the dataset. Default is 'entry/data'.
    data_type : str, optional
        Output data type. Default is 'float32'.
    crop : tuple of int, optional
        Crop images by (top, bottom, left, right).
    chunks : tuple of int, optional
        Chunk shape in (width, height, nrow, ncol) order.
        Chunk dimensions are capped at the dataset dimensions.
    pattern : str, optional
        Filter TIFF filenames by a pattern, e.g. 'img'.
    overwrite : bool, optional
        Overwrite an existing output file. Default is True.
    options : dict, optional
        Add metadata, e.g. options={"entry/energy": 53}.

    Returns
    -------
    pathlib.Path
        Path to the HDF5 file.

    Notes
    -----
    Each TIFF is one (height, width) image in the logical array
    (nrow, ncol, height, width). Its flattened scan-position index is
    row * ncol + column. The scan_order parameter maps acquisition
    order to this index; it does not change the output axis order.

    Images are cropped and transposed when written. The stored dataset
    has shape (cropped_width, cropped_height, nrow, ncol).
    """
    for name, value in (("nrow", nrow), ("ncol", ncol)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (
        int, np.integer)) or value < 1):
            raise ValueError(f"{name} must be a positive integer.")

    if scan_order not in ("row", "column", "snake_row", "snake_column"):
        raise ValueError("scan_order must be 'row', 'column', "
                         "'snake_row', or 'snake_column'.")

    for name, values, minimum in (("crop", crop, 0), ("chunks", chunks, 1)):
        if len(values) != 4 or any(isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum
                                   for value in values):
            raise ValueError(f"{name} must contain four integers >= {minimum}.")

    for metadata in options.values():
        if not isinstance(metadata, dict):
            raise TypeError("Each metadata option must be a dictionary.")

    input_path = Path(input_path)
    file_pattern = "*.tif*" if pattern is None else f"*{pattern}*.tif*"
    list_file = sorted(losa.find_file(str(input_path / file_pattern)))

    if len(list_file) != nrow * ncol:
        raise ValueError(
            f"Expected {nrow * ncol} TIFF files, found {len(list_file)}.")

    first_image = losa.load_image(list_file[0])
    if first_image.ndim != 2:
        raise ValueError("Input images must be 2D.")

    height, width = first_image.shape
    cr_top, cr_bottom, cr_left, cr_right = crop
    cr_height = height - cr_top - cr_bottom
    cr_width = width - cr_left - cr_right

    if cr_height < 1 or cr_width < 1:
        raise ValueError("Can't crop images with the given parameters.")

    # Map acquisition order to flattened scan-position indices.
    scan_indices = np.arange(nrow * ncol).reshape(nrow, ncol)

    if scan_order == "row":
        acquisition_indices = scan_indices.ravel()
    elif scan_order == "column":
        acquisition_indices = scan_indices.T.ravel()
    elif scan_order == "snake_row":
        scan_indices[1::2] = scan_indices[1::2, ::-1].copy()
        acquisition_indices = scan_indices.ravel()
    else:
        scan_indices[:, 1::2] = scan_indices[::-1, 1::2].copy()
        acquisition_indices = scan_indices.T.ravel()

    output_path = Path(output_path).resolve()
    if output_path.suffix.lower() not in {".hdf", ".h5", ".nxs", ".hdf5"}:
        output_path = output_path.with_suffix(".hdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_shape = (cr_width, cr_height, nrow, ncol)
    chunks = tuple(min(c, s) for c, s in zip(chunks, data_shape))

    with h5py.File(output_path, "w" if overwrite else "x") as hdf:
        data_out = hdf.create_dataset(
            key_path, shape=data_shape, dtype=data_type, chunks=chunks)

        for metadata in options.values():
            for key, value in metadata.items():
                hdf.create_dataset(key, data=value)

        for index, (file_path, flat_index) in enumerate(
            zip(list_file, acquisition_indices)):
            image = first_image if index == 0 else losa.load_image(file_path)

            if image.shape != (height, width):
                raise ValueError(
                    f"Image shape mismatch: {file_path}. "
                    f"Expected {(height, width)}, got {image.shape}.")

            row, col = divmod(int(flat_index), ncol)
            data_out[:, :, row, col] = image[
                cr_top:cr_top + cr_height,
                cr_left:cr_left + cr_width].T
            if index % 20 == 0:
                print(f" Done update  {index} file!")

    return output_path


convert_tif_scan_to_hdf(folder_path, output_folder, num_row, num_column,
                        scan_order="row", key_path="entry/data",
                        data_type="float32", crop=(0, 0, 0, 0),
                        chunks=chunks, pattern=None,
                        overwrite=True)
