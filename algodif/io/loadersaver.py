# ============================================================================
# ============================================================================
# Copyright (c) 2026 Nghia T. Vo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Author: Nghia T. Vo
# E-mail:
# Description: Python module for loading and saving data.
# Contributors:
# ============================================================================

"""
Module for I/O tasks:

    -   Load/save data from/to an image file, csv, dat, json or a hdf/nxs file.
    -   Get file names, make file/folder name.
"""

import csv
import json
import platform
from pathlib import Path
from fnmatch import fnmatchcase
from collections import deque
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


PIPE = "│"
ELBOW = "└──"
TEE = "├──"
PIPE_PREFIX = "│   "
SPACE_PREFIX = "    "


def __correct_path(file_path):
    """
    Correct escaped sequences in WinOS file path.
    """
    if isinstance(file_path, Path):
        file_path = str(file_path)
    escape_sequences = {
        "\a": r"\a",
        "\b": r"\b",
        "\f": r"\f",
        "\n": r"\n",
        "\r": r"\r",
        "\t": r"\t",
        "\v": r"\v",
    }
    for char, escaped in escape_sequences.items():
        if char in file_path:
            file_path = file_path.replace(char, escaped)
    file_path = file_path.replace("\\", "/")
    return Path(file_path)


def __get_path(file_path, check_exist=True):
    """
    Get/check a file path
    """
    if platform.system() == "Windows":
        file_path = __correct_path(file_path)
    else:
        file_path = Path(file_path)
    if check_exist:
        if not file_path.exists():
            raise ValueError(f"No such file: {file_path}")
    return file_path


def load_image(file_path):
    """
    Load an image and convert it to a 2D array

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    array_like
        2D array.
    """
    try:
        mat = np.array(Image.open(__get_path(file_path)), dtype=np.float32)
    except Exception as error:
        raise ValueError(error)
    if len(mat.shape) > 2:
        axis_m = np.argmin(mat.shape)
        mat = np.mean(mat, axis=axis_m)
    return mat


def get_hdf_information(file_path, display=False):
    """
    Get information of datasets in a hdf/nxs file.

    Parameters
    ----------
    file_path : str
        Path to the file.
    display : bool
        Print the results onto the screen if True.

    Returns
    -------
    list_key : str
        Keys to the datasets.
    list_shape : tuple of int
        Shapes of the datasets.
    list_type : str
        Types of the datasets.
    """
    hdf_object = h5py.File(__get_path(file_path), 'r')
    keys = []
    hdf_object.visit(keys.append)
    list_key, list_shape, list_type = [], [], []
    for key in keys:
        try:
            data = hdf_object[key]
            if isinstance(data, h5py.Group):
                list_tmp = list(data.items())
                if list_tmp:
                    for key2, _ in list_tmp:
                        list_key.append(key + "/" + key2)
                else:
                    list_key.append(key)
            else:
                list_key.append(data.name)
        except KeyError:
            list_key.append(key)
            pass
    for i, key in enumerate(list_key):
        shape, dtype = None, None
        try:
            data = hdf_object[list_key[i]]
            if isinstance(data, h5py.Dataset):
                shape, dtype = data.shape, data.dtype
            list_shape.append(shape)
            list_type.append(dtype)
        except KeyError:
            list_shape.append(shape)
            list_type.append(dtype)
            pass
    hdf_object.close()
    if display:
        if list_key:
            for i, key in enumerate(list_key):
                print(key + " : " + str(list_shape[i]) + " : " + str(
                    list_type[i]))
        else:
            print("Empty file !!!")
    return list_key, list_shape, list_type


def find_hdf_key(file_path, pattern, display=False):
    """
    Find datasets matching the name-pattern in a hdf/nxs file.

    Parameters
    ----------
    file_path : str
        Path to the file.
    pattern : str
        Pattern to find the full names of the datasets.
    display : bool
        Print the results onto the screen if True.

    Returns
    -------
    list_key : str
        Keys to the datasets.
    list_shape : tuple of int
        Shapes of the datasets.
    list_type : str
        Types of the datasets.
    """
    hdf_object = h5py.File(__get_path(file_path), 'r')
    list_key, keys = [], []
    hdf_object.visit(keys.append)
    for key in keys:
        try:
            data = hdf_object[key]
            if isinstance(data, h5py.Group):
                list_tmp = list(data.items())
                if list_tmp:
                    for key2, _ in list_tmp:
                        list_key.append(key + "/" + key2)
                else:
                    list_key.append(key)
            else:
                list_key.append(data.name)
        except KeyError:
            pass
    list_dkey, list_dshape, list_dtype = [], [], []
    for _, key in enumerate(list_key):
        if pattern in key:
            list_dkey.append(key)
            shape, dtype = None, None
            try:
                data = hdf_object[key]
                if isinstance(data, h5py.Dataset):
                    shape, dtype = data.shape, data.dtype
                list_dtype.append(dtype)
                list_dshape.append(shape)
            except KeyError:
                list_dtype.append(dtype)
                list_dshape.append(shape)
                pass
    hdf_object.close()
    if display:
        if list_dkey:
            for i, key in enumerate(list_dkey):
                print(key + " : " + str(list_dshape[i]) + " : " + str(
                    list_dtype[i]))
        else:
            print("Can't find datasets with keys matching the "
                  "pattern: {}".format(pattern))
    return list_dkey, list_dshape, list_dtype


def _get_subgroups(hdf_object, key=None):
    """
    Supplementary method for building the tree view of a hdf5 file.
    Return the name of subgroups.
    """
    list_group = []
    if key is None:
        for group in hdf_object.keys():
            list_group.append(group)
        if len(list_group) == 1:
            key = list_group[0]
        else:
            key = ""
    else:
        if key in hdf_object:
            try:
                obj = hdf_object[key]
                if isinstance(obj, h5py.Group):
                    for group in hdf_object[key].keys():
                        list_group.append(group)
            except KeyError:
                pass
    if len(list_group) > 0:
        list_group = sorted(list_group)
    return list_group, key


def _add_branches(tree, hdf_object, key, key1, index, last_index, prefix,
                  connector, level, add_shape):
    """
    Supplementary method for building the tree view of a hdf5 file.
    Add branches to the tree.
    """
    shape = None
    key_comb = key + "/" + key1
    if add_shape is True:
        if key_comb in hdf_object:
            try:
                obj = hdf_object[key_comb]
                if isinstance(obj, h5py.Dataset):
                    shape = str(obj.shape)
            except KeyError:
                shape = str("-> ???External-link???")
    if shape is not None:
        tree.append(f"{prefix}{connector} {key1} {shape}")
    else:
        tree.append(f"{prefix}{connector} {key1}")
    if index != last_index:
        prefix += PIPE_PREFIX
    else:
        prefix += SPACE_PREFIX
    _make_tree_body(tree, hdf_object, prefix=prefix, key=key_comb,
                    level=level, add_shape=add_shape)


def _make_tree_body(tree, hdf_object, prefix="", key=None, level=0,
                    add_shape=True):
    """
    Supplementary method for building the tree view of a hdf5 file.
    Create the tree body.
    """
    entries, key = _get_subgroups(hdf_object, key)
    num_ent = len(entries)
    last_index = num_ent - 1
    level = level + 1
    if num_ent > 0:
        if last_index == 0:
            key = "" if level == 1 else key
            if num_ent > 1:
                connector = PIPE
            else:
                connector = ELBOW if level > 1 else ""
            _add_branches(tree, hdf_object, key, entries[0], 0, 0, prefix,
                          connector, level, add_shape)
        else:
            for index, key1 in enumerate(entries):
                connector = ELBOW if index == last_index else TEE
                if index == 0:
                    tree.append(prefix + PIPE)
                _add_branches(tree, hdf_object, key, key1, index, last_index,
                              prefix, connector, level, add_shape)


def get_hdf_tree(file_path, output=None, add_shape=True, display=True):
    """
    Get the tree view of a hdf/nxs file.

    Parameters
    ----------
    file_path : str
        Path to the file.
    output : str or None
        Path to the output file in a text-format file (.txt, .md,...).
    add_shape : bool
        Including the shape of a dataset to the tree if True.
    display : bool
        Print the tree onto the screen if True.

    Returns
    -------
    list of string
    """
    hdf_object = h5py.File(__get_path(file_path), 'r')
    tree = deque()
    _make_tree_body(tree, hdf_object, add_shape=add_shape)
    if output is not None:
        make_folder(output)
        output_file = open(output, mode="w", encoding="UTF-8")
        with output_file as stream:
            for entry in tree:
                print(entry, file=stream)
    else:
        if display:
            for entry in tree:
                print(entry)
    return tree


def load_hdf(file_path, key_path, return_file_obj=False):
    """
    Load a hdf/nexus dataset as an object.

    Parameters
    ----------
    file_path : str
        Path to the file.
    key_path : str
        Key path to the dataset.
    return_file_obj : bool, optional

    Returns
    -------
    objects
        hdf-dataset object, and file-object if return_file_obj is True.
    """
    try:
        hdf_object = h5py.File(__get_path(file_path), 'r')
    except Exception as error:
        raise ValueError(f"{error}")
    check = key_path in hdf_object
    if not check:
        raise ValueError(
            "Couldn't open object with the given key: {}".format(key_path))
    if return_file_obj:
        return hdf_object[key_path], hdf_object
    else:
        return hdf_object[key_path]


def make_folder(file_path):
    """
    Create a folder for saving file if the folder does not exist. This is a
    supplementary function for savers.

    Parameters
    ----------
    file_path : str
        Path to a file.
    """
    path = Path(file_path).resolve()
    if path.suffix:
        folder_path = path.parent
    else:
        folder_path = path
    if not folder_path.exists():
        try:
            folder_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Can't create: {folder_path}. Error: {e}")


def make_folder_name(folder_path, name_prefix="Output", zero_prefix=5):
    """
    Create a new folder name to avoid overwriting.
    E.g: Output_00001, Output_00002...

    Parameters
    ----------
    folder_path : str
        Path to the parent folder.
    name_prefix : str
        Name prefix
    zero_prefix : int
        Number of zeros to be added to file names.
    Returns
    -------
    str
        Name of the folder.
    """
    folder_path = Path(folder_path)
    scan_name_prefix = name_prefix + "_"
    existing_folders = list(folder_path.glob(f"{scan_name_prefix}*"))
    num_folder_exist = len(existing_folders)
    num_folder_new = num_folder_exist + 1
    name_tmp = f"{num_folder_new: 0{zero_prefix}}"
    scan_name = scan_name_prefix + name_tmp
    while (folder_path / scan_name).is_dir():
        num_folder_new += 1
        name_tmp = f"{num_folder_new: 0{zero_prefix}}"
        scan_name = scan_name_prefix + name_tmp
    return scan_name


def make_file_name(file_path):
    """
    Create a new file name to avoid overwriting.

    Parameters
    ----------
    file_path : str or Path object
        Path to the file.

    Returns
    -------
    str
        Updated file path.
    """
    file_path = Path(file_path)
    file_base = file_path.stem
    file_ext = file_path.suffix
    parent_dir = file_path.parent
    if file_path.exists():
        nfile = 0
        while True:
            name_add = f"_{nfile: 04d}"
            new_file_name = f"{file_base}{name_add}{file_ext}"
            new_file_path = parent_dir / new_file_name
            if new_file_path.exists():
                nfile += 1
            else:
                file_path = new_file_path
                break
    return str(file_path)


def find_file(path, case_sensitive=False):
    """
    Search file

    Parameters
    ----------
    path : str
        Path and pattern to find files. e.g. 'C:/data/*.tif*'
    case_sensitive : bool, optional
        If True, pattern matching is case-sensitive. Defaults to False.

    Returns
    -------
    str or list of str
        List of files.
    """
    path = __correct_path(path)
    directory = path.parent
    pattern = path.name
    if case_sensitive:
        file_paths = list(directory.glob(path.name))
    else:
        pattern_lower = pattern.casefold()
        file_paths = [p for p in directory.iterdir()
                      if fnmatchcase(p.name.casefold(), pattern_lower)]
    if not file_paths:
        raise FileNotFoundError(f"No files found matching: {path}")
    return sorted([file.as_posix() for file in file_paths])


def save_image(file_path, mat, overwrite=True):
    """
    Save a 2D array to an image.

    Parameters
    ----------
    file_path : str
        Path to the file.
    mat : int or float
        2D array.
    overwrite : bool
        Overwrite an existing file if True.

    Returns
    -------
    str
        Updated file path.
    """
    file_path = __get_path(file_path, check_exist=False)
    file_path = file_path.resolve()
    file_ext = file_path.suffix
    if file_ext not in (".tif", ".tiff"):
        mat = np.uint8(
            255.0 * (mat - np.min(mat)) / (np.max(mat) - np.min(mat)))
    else:
        data_type = str(mat.dtype)
        if "complex" in data_type:
            raise ValueError(f"Can't save to tiff with format: {data_type}")
    image = Image.fromarray(mat)
    if not overwrite:
        file_path = make_file_name(file_path)
    make_folder(file_path)
    try:
        image.save(file_path)
    except Exception as error:
        raise ValueError(f"Couldn't write to file: {file_path}. Error {error}")
    return file_path


def open_hdf_stream(file_path, data_shape, key_path='entry/data',
                    data_type='float32', overwrite=True, **options):
    """
    Write an array to a hdf/nxs file with options to add metadata.

    Parameters
    ----------
    file_path : str
        Path to the file.
    data_shape : tuple of int
        Shape of the data.
    key_path : str
        Key path to the dataset.
    data_type: str
        Type of data.
    overwrite : bool
        Overwrite the existing file if True.
    options : dict, optional
        Add metadata. E.g options={"entry/angles": angles, "entry/energy": 53}.

    Returns
    -------
    object
        hdf object.
    """
    file_path = __get_path(file_path, check_exist=False)
    file_path = file_path.resolve()
    if file_path.suffix.lower() not in {'.hdf', '.h5', '.nxs', '.hdf5'}:
        file_path = file_path.with_suffix('.hdf')
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        file_path = Path(make_file_name(file_path))
    try:
        ofile = h5py.File(file_path, 'w')
    except Exception as error:
        raise ValueError(f"Couldn't write to file: {file_path}. Error {error}")
    if len(options) != 0:
        for opt_name in options:
            opts = options[opt_name]
            for key in opts:
                if key_path in key:
                    msg = "!!!Selected key-path, '{0}', can not be a child " \
                          "key-path of '{1}'!!!\n!!!Change to make sure " \
                          "they are at the same level!!!".format(key, key_path)
                    raise ValueError(msg)
                ofile.create_dataset(key, data=opts[key])
    data_out = ofile.create_dataset(key_path, data_shape, dtype=data_type)
    return data_out


def load_dat(file_path, dtype=np.float32, shape=(192, 4096), skip_columns=2):
    """
    Load data from a dat text file into a NumPy array. Specific for GeRM
    detector output.

    Parameters
    ----------
    file_path : str or Path object
        Path to the file.
    dtype : data-type, optional
        Desired data type of the output array. Default is `np.float32`.
    shape : tuple of int, optional
        Desired shape to reshape the loaded data into. Default is (192, 4096).
    skip_columns : int, optional
        Number of columns to skip at the beginning of each line. Default is 2.

    Returns
    -------
    ndarray
        2D array.
    """
    file_path = __get_path(file_path)
    try:
        expected_size = np.prod(shape)
        with open(file_path, "r") as f:
            data = []
            for line in f:
                values = line.split()[skip_columns:]
                data.extend(values)
        data = np.array(data, dtype=dtype)
        current_size = data.size
        if current_size < expected_size:
            missing = expected_size - current_size
            print(f"Warning: {missing} missing values detected. "
                  f"Padding with zeros.")
            data = np.pad(data, (0, missing), mode="constant")
        elif current_size > expected_size:
            extra = current_size - expected_size
            print(f"Warning: {extra} extra values detected. Truncating.")
            data = data[:expected_size]
        return data.reshape(shape)
    except Exception as e:
        raise ValueError(f"An error occurred while loading {file_path}: {e}")


def load_d_spacing_from_pearson_crystal_data_file(file_path, num_use=12,
                                                  threshold=35, norm=True,
                                                  check=True):
    """
    Load and select d-spacing values from a Pearson crystal data CSV file.

    Parameters
    ----------
    file_path : str or Path object
        Path to the Pearson crystal data CSV file.
    num_use : int, optional
        Maximum number of reflections to return (default: 12).
    threshold : float, optional
        Minimum intensity threshold for peak selection (default: 35).
    norm : bool, optional
        Normalize intensities to their maximum if True (default: True).
    check : bool, optional
        Print output if True (default: True).

    Returns
    -------
    ndarray of shape (N, 2)
        [d_spacing (Angstrom), intensity] pairs sorted by decreasing d-spacing.
    """
    material_data = load_csv(__get_path(file_path))
    d_spacing_values = material_data[1:, 1] * 10
    intensities = material_data[1:, 3]
    mask = intensities > threshold
    d_spacing_sel = d_spacing_values[mask]
    intensity_sel = intensities[mask]
    stacked = np.column_stack((d_spacing_sel, intensity_sel))
    unique_rows = np.unique(stacked, axis=0)
    d_spacing_values = unique_rows[:, 0][::-1]
    intensities = unique_rows[:, 1][::-1]
    npoint = len(d_spacing_values)
    if num_use < npoint:
        d_spacing_values = d_spacing_values[:num_use]
        intensities = intensities[:num_use]
    if norm:
        intensities = intensities / np.max(intensities)
    q_values = 2 * np.pi / d_spacing_values
    if check:
        for i, d in enumerate(d_spacing_values):
            print(
                f"{i + 1: 2}. q-value {q_values[i]: 0.3f}. "
                f"d-spacing: {d: 0.3f}. Intensity {intensities[i]: 0.3f}")
    return np.asarray(list(zip(d_spacing_values, intensities)))


def load_json(file_path):
    """
    Load a json file.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    dict
        Dictionary.
    """
    with open(__get_path(file_path), 'r', encoding='utf-8') as file:
        data_dict = json.load(file)
    return data_dict


def find_key_of_list_json(file_path_or_dict):
    """
    Find keys whose values are non-empty lists of numbers.
    Useful for identifying motor names associated with lists of recorded
    positions in JSON data.

    Parameters
    ----------
    file_path_or_dict : str, pathlib.Path or dict
        JSON file path or dictionary to search.

    Returns
    -------
    list of str
        Matching keys in depth-first order.
    """
    if isinstance(file_path_or_dict, (str, Path)):
        with open(file_path_or_dict, "r", encoding="utf-8") as file:
            data = json.load(file)
    elif isinstance(file_path_or_dict, dict):
        data = file_path_or_dict
    else:
        raise TypeError("file_path must be a JSON file path or dictionary.")
    result = []

    def is_list_of_numbers(value):
        return isinstance(value, list) and all(
            isinstance(item, (int, float)) for item in value)

    def __search(d):
        for key, value in d.items():
            if is_list_of_numbers(value):
                result.append(key)
            elif isinstance(value, dict):
                __search(value)

    __search(data)
    return result


def get_list_by_key_json(file_path, partial_key, return_all=False):
    """
    Find list values whose keys contain a given substring.
    Useful for getting motor positions of a 1D scan.

    Parameters
    ----------
    file_path : str, pathlib.Path or dict
        JSON file path, Path object, or dictionary
    partial_key : str
        Case-sensitive substring to match against keys.
    return_all : bool, optional
        Return all matches instead of the first. Default is False.

    Returns
    -------
    list or None
        First matching list, or None if no match is found.
        If return_all is True, return a list of matching lists, or an empty
        list.
    """
    if not isinstance(partial_key, str):
        raise TypeError("partial_key must be a string.")
    if not isinstance(return_all, bool):
        raise TypeError("return_all must be a boolean.")

    if isinstance(file_path, (str, Path)):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    elif isinstance(file_path, dict):
        data = file_path
    else:
        raise TypeError("file_path must be a JSON file path or dictionary.")

    def __search(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if (isinstance(key, str) and partial_key in key
                        and isinstance(value, list)):
                    yield value
                if isinstance(value, (dict, list)):
                    yield from __search(value)
        elif isinstance(node, list):
            for item in node:
                yield from __search(item)

    matches = __search(data)
    if not return_all:
        return next(matches, None)
    results = list(matches)
    return results[0] if len(results) == 1 else results


def load_csv(file_path, dtype=float, delimiter=","):
    """
    Load data from a CSV file into a NumPy array.

    Parameters
    ----------
    file_path : str or Path object
        Path to the file.
    dtype : data-type, optional
        Data type of the output array. Default is float.
    delimiter : str, optional
        The string used to separate values. Default is ','.

    Returns
    -------
    numpy.ndarray
        The loaded data as a NumPy array.
    """
    file_path = __get_path(file_path)
    try:
        data = np.genfromtxt(file_path, delimiter=delimiter, dtype=dtype)
    except Exception as e:
        raise IOError(f"Failed to load CSV file at {file_path}. Error: {e}")
    return data


def save_csv(file_path, data):
    """
    Save a 2D list or a NumPy array to a CSV file.

    Creates the folder if it does not exist. If the file already exists,
    it will be overwritten.

    Parameters
    ----------
    file_path : str or Path object
        Path to the file.
    data : list of lists or numpy.ndarray
        The 2D data to be saved into a CSV file.
    """
    file_path = __get_path(file_path, check_exist=False)
    make_folder(file_path)
    if isinstance(data, np.ndarray):
        data = data.tolist()
    elif not isinstance(data, list):
        raise ValueError("Input data must be a list of lists or a numpy array")
    try:
        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)
    except Exception as e:
        raise IOError(f"Failed to save CSV file at {file_path}. Error: {e}")


def save_image_plt(file_path, mat, x_label="X", y_label="Y", x_max=None,
                   y_max=None, aspect="auto", cmap=None, dpi=100, text_size=20,
                   font_weight="bold", color_bar=True, fig_size=(16, 16),
                   layout=(0.0, 0.02, 1.0, 1.0)):
    """
    Save a 2D array as an image using Matplotlib.

    Parameters
    ----------
    file_path : str or Path object
        Path to the file.
    mat : numpy.ndarray
        2D array.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    x_max : float, optional
        Maximum x-axis value. Default is the width of the image.
    y_max : float, optional
        Maximum y-axis value. Default is the height of the image.
    aspect : {'equal', 'auto'}
        The aspect ratio of the Axes. Default is 'auto'.
    cmap : str or matplotlib colormap, optional
        Colormap to use for displaying the image.
    dpi : int, optional
        Dots per inch for the output image. Default is 100.
    text_size : int, optional
        Font size for axis labels. Default is 20.
    font_weight : str, optional
        Font weight for labels. E.g., 'normal', 'bold'. Default is 'bold'.
    color_bar : bool, optional
        Whether to display a color bar. Default is True.
    fig_size : tuple of float, optional
        Figure size in inches as (width, height). Default is (16, 16).
    layout : tuple of float, optional
        The rect parameter for tight layout (left, bottom, right, top).
        Default is (0.0, 0.02, 1.0, 1.0).

    Returns
    -------
    str
        File path to the saved image.
    """
    file_path = __get_path(file_path, check_exist=False)
    height, width = mat.shape
    if x_max is None:
        x_max = width
    if y_max is None:
        y_max = height
    make_folder(file_path)
    original_rc = plt.rcParams.copy()
    plt.rcParams["font.size"] = text_size
    plt.rcParams["font.weight"] = font_weight
    plt.figure(dpi=dpi, figsize=fig_size)
    plt.margins(x=0.0, y=0.0)
    plt.tight_layout(rect=layout)
    plt.xlabel(x_label, fontsize=text_size, fontweight=font_weight)
    plt.ylabel(y_label, fontsize=text_size, fontweight=font_weight)
    imshow_args = {"extent": [0, x_max, y_max, 0], "aspect": aspect}
    if cmap is not None:
        imshow_args["cmap"] = cmap
    plt.imshow(mat, **imshow_args)
    if color_bar:
        plt.colorbar(shrink=0.4)
    try:
        plt.savefig(file_path, dpi=dpi)
    except IOError as e:
        print(f"Couldn't write to file {file_path}")
        raise e
    finally:
        plt.close()
        plt.rcParams.update(original_rc)
    return str(file_path)


def save_plot(file_path, y_list, x_list=None, x_lim=None, y_lim=None,
              x_label="Channel", y_label="Intensity", marker="o", color="red",
              linestyle="--", font_size=12, font_weight="bold",
              fig_size=(8.0, 7.0), dpi=200, layout=(0.02, 0.0, 1.0, 1.0)):
    """
    Save a line plot using Matplotlib.

    Parameters
    ----------
    file_path : str or Path
        Output path for saving the image. Must use Unix-style ('/') separators.
    y_list : array-like
        Y-axis data to plot.
    x_list : array-like, optional
        X-axis data to plot. If None, uses index values.
    x_lim : tuple, optional
        Limits for the x-axis as (min, max).
    y_lim : tuple, optional
        Limits for the y-axis as (min, max).
    x_label : str, optional
        Label for the x-axis. Default is "Channel".
    y_label : str, optional
        Label for the y-axis. Default is "Intensity".
    marker : str, optional
        Marker style for data points. Default is 'o'.
    linestyle : str, optional
        Line style (e.g., '-', '--', '-.', ':'). Default is '--'.
    color : str, optional
        Color for the plot. Default is 'red'.
    font_size : int, optional
        Font size for axis labels. Default is 12.
    font_weight : str, optional
        Font weight for axis labels ('bold', 'normal',...). Default is 'bold'.
    fig_size : tuple of float, optional
        Figure size in inches as (width, height). Default is (8.0, 7.0).
    dpi : int, optional
        Dots per inch for the saved image. Default is 200.
    layout : tuple of float, optional
        Rect layout for tight_layout. Default is (0.02, 0.0, 1.0, 1.0).

    Returns
    -------
    str
        The file path where the plot was saved.

    Raises
    ------
    ValueError
        If the file path contains Windows-style backslashes.
    IOError
        If saving the figure fails.
    """
    file_path = __get_path(file_path, check_exist=False)
    make_folder(file_path)
    original_rc = plt.rcParams.copy()
    plt.rcParams["font.size"] = font_size
    plt.rcParams["font.weight"] = font_weight
    plt.figure(dpi=dpi, figsize=fig_size)
    plt.margins(x=0.02, y=0.02)
    plt.tight_layout(rect=layout)
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    if x_list is None:
        plt.plot(y_list, color=color, marker=marker, linestyle=linestyle)
    else:
        plt.plot(x_list, y_list, color=color, marker=marker,
                 linestyle=linestyle)
    plt.xlabel(x_label, fontsize=font_size, fontweight=font_weight)
    plt.ylabel(y_label, fontsize=font_size, fontweight=font_weight)
    try:
        plt.savefig(file_path, dpi=dpi)
    except IOError as e:
        print(f"Couldn't write to file {file_path}")
        raise e
    finally:
        plt.close()
        plt.rcParams.update(original_rc)
    return str(file_path)


def save_q_plot(file_path, q_list, list_data, q_standard=None, start=0,
                stop=None, x_ticks=None, y_scale="log", y_lim=None,
                fig_size=(17.0, 9.0), dpi=100, font_size=12,
                font_weight="bold", layout=(0.0, 0.0, 1.01, 0.98),
                line_style="-", marker=None):
    """
    Save a Q-space plot with optional standard lines and log scale.

    Parameters
    ----------
    file_path : str or Path
        Output file path.
    q_list : array-like
        Q values (x-axis).
    list_data : array-like
        Intensity values (y-axis).
    q_standard : array-like, optional
        Vertical lines to mark standard Q values.
    start : int, optional
        Start index to slice data. Default is 0.
    stop : int, optional
        Stop index to slice data. Default is end.
    x_ticks : array-like, optional
        Custom x-tick positions.
    y_scale : str, optional
        Y-axis scale, 'log' or 'linear'. Default is 'log'.
    y_lim : tuple, optional
        Y-axis limits.
    fig_size : tuple, optional
        Figure size in inches. Default is (17, 9).
    dpi : int, optional
        Dots per inch for the output image. Default is 100.
    font_size : int, optional
        Font size for labels. Default is 12.
    font_weight : str, optional
        Font weight for labels. Default is 'bold'.
    layout : tuple, optional
        Layout rect for tight_layout. Default is (0.0, 0.0, 1.01, 0.98).
    line_style : str, optional
        Line style for data curve. Default is '-'.
    marker : str or None, optional
        Marker style. Default is None.

    Returns
    -------
    str
        Path where the figure is saved.
    """
    file_path = __get_path(file_path, check_exist=False)
    if stop is None:
        stop = len(list_data)
    q_list = q_list[start:stop]
    list_data = list_data[start:stop]
    original_rc = plt.rcParams.copy()
    plt.rcParams["font.size"] = font_size
    plt.rcParams["font.weight"] = font_weight
    plt.figure(dpi=dpi, figsize=fig_size)
    plt.margins(x=0.01, y=0.01)
    plt.tight_layout(rect=layout)
    if x_ticks is None:
        q_start = np.floor(np.min(q_list))
        q_stop = np.ceil(np.max(q_list))
        q_step = (q_stop - q_start) / 20
        x_ticks = np.arange(q_start, q_stop, q_step)
    plt.xticks(x_ticks)
    plt.yscale(y_scale)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.plot(q_list, list_data, linestyle=line_style, marker=marker,
             color="green", label="Measured")
    if q_standard is not None:
        q_max = np.max(list_data)
        plt.vlines(q_standard, ymin=0, ymax=q_max,
                   linestyles="solid", color="red", label="Theory")
    plt.grid(True)
    make_folder(file_path)
    plt.savefig(file_path, dpi=dpi)
    plt.close()
    plt.rcParams.update(original_rc)
    return str(file_path)
