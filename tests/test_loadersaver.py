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
# ============================================================================
# Contributors:
# ============================================================================

"""
Tests for methods in loadersaver.py
"""

import os
import json
import shutil
import platform
import unittest
from collections import deque
from pathlib import Path
import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import algodif.io.loadersaver as losa


class LoaderSaverMethods(unittest.TestCase):
    tmp_root = "./tmp_loadersaver"
    tmp_dir = tmp_root + "/data"

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(cls.tmp_dir):
            os.makedirs(cls.tmp_dir)

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.tmp_root):
            shutil.rmtree(cls.tmp_root, ignore_errors=True)

    def _make_hdf(self, name):
        file_path = f"{self.tmp_dir}/{name}"
        with h5py.File(file_path, "w") as ifile:
            ifile.create_dataset("entry/data",
                                 data=np.float32(np.random.rand(64, 64)))
            ifile.create_dataset("entry/energy", data=25.0)
            ifile.create_group("entry/position/stage")
        return file_path

    def test_correct_path(self):
        correct_path = getattr(losa, "__correct_path")
        self.assertEqual(correct_path("C:\\data\\img.tif"),
                         Path("C:/data/img.tif"))
        self.assertEqual(correct_path("C:\\data\test.tif"),
                         Path("C:/data/test.tif"))
        self.assertEqual(correct_path(Path("./tmp/img.tif")),
                         Path("./tmp/img.tif"))

    def test_get_path(self):
        get_path = getattr(losa, "__get_path")
        file_path = f"{self.tmp_dir}/get_path.txt"
        with open(file_path, "w") as f:
            f.write("test")
        self.assertTrue(isinstance(get_path(file_path), Path))
        self.assertTrue(isinstance(
            get_path(f"{self.tmp_dir}/not_exist.txt", check_exist=False),
            Path))
        self.assertRaises(ValueError, get_path,
                          f"{self.tmp_dir}/not_exist.txt")

    def test_load_image(self):
        file_path = f"{self.tmp_dir}/img.tif"
        mat_in = np.float32(np.random.rand(64, 48))
        losa.save_image(file_path, mat_in)
        mat = losa.load_image(file_path)
        self.assertTrue(mat.shape == (64, 48))
        self.assertTrue(np.allclose(mat, mat_in))

        file_path = f"{self.tmp_dir}/img_rgb.png"
        losa.save_image(file_path, np.float32(np.random.rand(64, 64, 3)))
        mat = losa.load_image(file_path)
        self.assertTrue(mat.shape == (64, 64))

        self.assertRaises(ValueError, losa.load_image,
                          f"{self.tmp_dir}/img_not_exist.png")

    def test_get_hdf_information(self):
        file_path = self._make_hdf("info.hdf")
        list_key, list_shape, list_type = losa.get_hdf_information(file_path)
        self.assertTrue(len(list_key) == len(list_shape) == len(list_type))
        self.assertTrue(isinstance(list_key[0], str))
        info = {key.lstrip("/"): (shape, dtype) for key, shape, dtype
                in zip(list_key, list_shape, list_type)}
        self.assertTrue(info["entry/data"][0] == (64, 64))
        self.assertTrue(info["entry/data"][1] == np.float32)
        self.assertTrue(info["entry/energy"][0] == ())
        self.assertTrue(info["entry/position/stage"][0] is None)

        self.assertRaises(ValueError, losa.get_hdf_information,
                          f"{self.tmp_dir}/info_not_exist.hdf")

    def test_find_hdf_key(self):
        file_path = self._make_hdf("find_key.hdf")
        list_key, list_shape, list_type = losa.find_hdf_key(file_path,
                                                            "entry/energy")
        self.assertTrue(len(list_key) > 0)
        self.assertTrue(all("entry/energy" in key for key in list_key))
        self.assertTrue(len(list_key) == len(list_shape) == len(list_type))

        list_key, list_shape, _ = losa.find_hdf_key(file_path, "data")
        self.assertTrue(all("data" in key for key in list_key))
        self.assertTrue(list_shape[0] == (64, 64))

        list_key, _, _ = losa.find_hdf_key(file_path, "stage")
        self.assertTrue(list_key[0].lstrip("/") == "entry/position/stage")

        results = losa.find_hdf_key(file_path, "no_such_key")
        self.assertTrue(results == ([], [], []))

    def test_get_hdf_tree(self):
        file_path = self._make_hdf("tree.hdf")
        tree = losa.get_hdf_tree(file_path, display=False)
        self.assertTrue(isinstance(tree, deque))
        text = "\n".join(tree)
        self.assertTrue("entry" in text)
        self.assertTrue("data (64, 64)" in text)
        self.assertTrue("energy" in text)
        self.assertTrue("stage" in text)

        tree = losa.get_hdf_tree(file_path, add_shape=False, display=False)
        self.assertFalse("(64, 64)" in "\n".join(tree))

        output = f"{self.tmp_dir}/tree_out/tree.txt"
        tree = losa.get_hdf_tree(file_path, output=output)
        self.assertTrue(os.path.isfile(output))
        with open(output, "r", encoding="UTF-8") as f:
            lines = f.read().splitlines()
        self.assertTrue(len(lines) == len(tree))

        self.assertRaises(ValueError, losa.get_hdf_tree,
                          f"{self.tmp_dir}/tree_not_exist.hdf")

    def test_load_hdf(self):
        file_path = self._make_hdf("load.hdf")
        data = losa.load_hdf(file_path, "entry/data")
        self.assertTrue(isinstance(data, h5py.Dataset)
                        and data.shape == (64, 64))
        data.file.close()

        data, hdf_object = losa.load_hdf(file_path, "entry/data",
                                         return_file_obj=True)
        self.assertTrue(isinstance(hdf_object, h5py.File))
        self.assertTrue(data[:].shape == (64, 64))
        hdf_object.close()

        self.assertRaises(ValueError, losa.load_hdf, file_path,
                          "entry/data1")
        self.assertRaises(ValueError, losa.load_hdf,
                          f"{self.tmp_dir}/load_not_exist.hdf", "entry/data")

    def test_make_folder(self):
        folder = f"{self.tmp_dir}/make_folder/sub1/sub2"
        losa.make_folder(folder)
        self.assertTrue(os.path.isdir(folder))

        file_path = f"{self.tmp_dir}/make_folder/sub3/file.txt"
        losa.make_folder(file_path)
        self.assertTrue(os.path.isdir(os.path.dirname(file_path)))
        self.assertFalse(os.path.exists(file_path))

        losa.make_folder(file_path)
        self.assertTrue(os.path.isdir(os.path.dirname(file_path)))

    def test_make_folder_name(self):
        parent = f"{self.tmp_dir}/folder_name"
        os.makedirs(parent, exist_ok=True)
        name1 = losa.make_folder_name(parent)
        self.assertEqual(name1, "Output_00001")

        os.makedirs(os.path.join(parent, name1))
        name2 = losa.make_folder_name(parent)
        self.assertEqual(name2, "Output_00002")

        # Count + 1 collides with an existing folder, so the loop advances.
        os.makedirs(os.path.join(parent, name2))
        os.rmdir(os.path.join(parent, name1))
        name3 = losa.make_folder_name(parent)
        self.assertEqual(name3, "Output_00003")

        name4 = losa.make_folder_name(parent, name_prefix="Scan",
                                      zero_prefix=3)
        self.assertEqual(name4, "Scan_001")

    def test_make_file_name(self):
        file_path = f"{self.tmp_dir}/file_name.txt"
        self.assertEqual(losa.make_file_name(file_path),
                         str(Path(file_path)))

        with open(file_path, "w") as f:
            f.write("test")
        path1 = losa.make_file_name(file_path)
        self.assertEqual(Path(path1).name, "file_name_0000.txt")
        self.assertFalse(os.path.exists(path1))

        with open(path1, "w") as f:
            f.write("test")
        path2 = losa.make_file_name(file_path)
        self.assertEqual(Path(path2).name, "file_name_0001.txt")

    def test_find_file(self):
        folder = f"{self.tmp_dir}/find_file"
        os.makedirs(folder, exist_ok=True)
        for name in ("a_001.tif", "b_002.tif", "C_003.TIF", "d.txt"):
            with open(f"{folder}/{name}", "w") as f:
                f.write("test")
        files = losa.find_file(f"{folder}/*.tif")
        self.assertEqual([os.path.basename(f) for f in files],
                         sorted(["a_001.tif", "b_002.tif", "C_003.TIF"]))
        self.assertTrue(all(isinstance(f, str) for f in files))

        files = losa.find_file(f"{folder}/*.txt")
        self.assertEqual([os.path.basename(f) for f in files], ["d.txt"])

        self.assertRaises(FileNotFoundError, losa.find_file,
                          f"{folder}/*.png")

    @unittest.skipIf(platform.system() == "Windows",
                     "Path.glob is case-insensitive on Windows")
    def test_find_file_case_sensitive(self):
        folder = f"{self.tmp_dir}/find_file_case"
        os.makedirs(folder, exist_ok=True)
        for name in ("a_001.tif", "C_003.TIF"):
            with open(f"{folder}/{name}", "w") as f:
                f.write("test")
        files = losa.find_file(f"{folder}/*.tif", case_sensitive=True)
        self.assertEqual([os.path.basename(f) for f in files], ["a_001.tif"])

    def test_save_image(self):
        mat1 = np.float32(np.random.rand(64, 64))
        mat2 = np.float32(np.random.rand(64, 64, 3))
        file_path = f"{self.tmp_dir}/save_image/img.tif"
        losa.save_image(file_path, mat1)
        self.assertTrue(os.path.isfile(file_path))

        file_path = f"{self.tmp_dir}/save_image/img.png"
        losa.save_image(file_path, mat2)
        self.assertTrue(os.path.isfile(file_path))

        path = losa.save_image(file_path, mat1, overwrite=False)
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(str(path) != str(Path(file_path).resolve()))

        mat_complex = np.complex64(np.random.rand(64, 64))
        self.assertRaises(ValueError, losa.save_image,
                          f"{self.tmp_dir}/save_image/img_complex.tif",
                          mat_complex)

    def test_open_hdf_stream(self):
        file_path = f"{self.tmp_dir}/stream/data.hdf"
        data_out = losa.open_hdf_stream(file_path, (64, 64))
        self.assertTrue(isinstance(data_out, h5py.Dataset))
        self.assertTrue(data_out.shape == (64, 64))
        self.assertTrue(data_out.dtype == np.float32)
        data_out.file.close()
        self.assertTrue(os.path.isfile(file_path))

        data_out = losa.open_hdf_stream(file_path, (64, 64), overwrite=False)
        new_path = data_out.file.filename
        data_out.file.close()
        self.assertTrue(Path(new_path) != Path(file_path).resolve())
        self.assertTrue(os.path.isfile(new_path))

        file_path = f"{self.tmp_dir}/stream/data2"
        data_out = losa.open_hdf_stream(file_path, (8, 8), key_path="a/b",
                                        data_type="int16")
        self.assertTrue(data_out.name == "/a/b" and
                        data_out.dtype == np.int16)
        data_out.file.close()
        self.assertTrue(os.path.isfile(file_path + ".hdf"))

        file_path = f"{self.tmp_dir}/stream/data3.hdf"
        data_out = losa.open_hdf_stream(file_path, (8, 8),
                                        options={"entry/energy": 25.0})
        hdf_file = data_out.file
        self.assertTrue(hdf_file["entry/energy"][()] == 25.0)
        hdf_file.close()

        self.assertRaises(ValueError, losa.open_hdf_stream,
                          f"{self.tmp_dir}/stream/data4.hdf", (8, 8),
                          options={"energy/entry/data": 25.0})

    def test_load_dat(self):
        file_path = f"{self.tmp_dir}/test.dat"
        with open(file_path, "w") as f:
            f.write("0 1 1.0 2.0 3.0\n")
            f.write("0 1 4.0 5.0 6.0\n")
        mat = losa.load_dat(file_path, shape=(2, 3), skip_columns=2)
        self.assertTrue(mat.shape == (2, 3) and mat.dtype == np.float32)
        self.assertTrue(np.array_equal(mat, [[1, 2, 3], [4, 5, 6]]))

        mat = losa.load_dat(file_path, shape=(3, 3), skip_columns=2)
        self.assertTrue(mat.shape == (3, 3))
        self.assertTrue(np.array_equal(mat[2], [0, 0, 0]))

        mat = losa.load_dat(file_path, shape=(1, 4), skip_columns=2)
        self.assertTrue(np.array_equal(mat, [[1, 2, 3, 4]]))

        bad_path = f"{self.tmp_dir}/bad.dat"
        with open(bad_path, "w") as f:
            f.write("0 1 abc def\n")
        self.assertRaises(ValueError, losa.load_dat, bad_path, shape=(1, 2))
        self.assertRaises(ValueError, losa.load_dat,
                          f"{self.tmp_dir}/not_exist.dat")

    def test_load_d_spacing_from_pearson_crystal_data_file(self):
        f_alias = losa.load_d_spacing_from_pearson_crystal_data_file
        file_path = f"{self.tmp_dir}/pearson.csv"
        with open(file_path, "w") as f:
            f.write("h,d,x,I\n")
            f.write("1,0.30,0,100\n")
            f.write("2,0.20,0,50\n")
            f.write("3,0.10,0,20\n")
            f.write("4,0.15,0,80\n")
        results = f_alias(file_path, check=False)
        self.assertTrue(results.shape == (3, 2))
        self.assertTrue(np.allclose(results[:, 0], [3.0, 2.0, 1.5]))
        self.assertTrue(np.allclose(results[:, 1], [1.0, 0.5, 0.8]))

        results = f_alias(file_path, norm=False, check=False)
        self.assertTrue(np.allclose(results[:, 1], [100, 50, 80]))

        results = f_alias(file_path, num_use=2, check=False)
        self.assertTrue(results.shape == (2, 2))

        results = f_alias(file_path, threshold=60, check=False)
        self.assertTrue(np.allclose(results[:, 0], [3.0, 1.5]))

        self.assertRaises(ValueError, f_alias,
                          f"{self.tmp_dir}/pearson_not_exist.csv")

    def test_load_json(self):
        file_path = f"{self.tmp_dir}/test.json"
        data_in = {"energy": 25.0, "motors": {"x": [1, 2, 3]}}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_in, f)
        self.assertEqual(losa.load_json(file_path), data_in)
        self.assertRaises(ValueError, losa.load_json,
                          f"{self.tmp_dir}/json_not_exist.json")

    def test_find_key_of_list_json(self):
        data = {"motors": {"x": [1, 2, 3], "y": [0.5, 1.5], "name": "abc",
                           "tags": ["a", "b"]},
                "energy": 25, "scan": [1, 2]}
        self.assertEqual(losa.find_key_of_list_json(data),
                         ["x", "y", "scan"])

        file_path = f"{self.tmp_dir}/key_list.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        self.assertEqual(losa.find_key_of_list_json(file_path),
                         ["x", "y", "scan"])
        self.assertEqual(losa.find_key_of_list_json(Path(file_path)),
                         ["x", "y", "scan"])

        self.assertEqual(losa.find_key_of_list_json({"a": 1}), [])
        self.assertRaises(TypeError, losa.find_key_of_list_json, 10)

    def test_get_list_by_key_json(self):
        f_alias = losa.get_list_by_key_json
        data = {"motors": {"x_pos": [1, 2, 3], "y_pos": [4, 5]},
                "other": {"x_pos_2": [7]}, "energy": 25}
        self.assertEqual(f_alias(data, "x_pos"), [1, 2, 3])
        self.assertEqual(f_alias(data, "x_pos", return_all=True),
                         [[1, 2, 3], [7]])
        self.assertEqual(f_alias(data, "y_pos", return_all=True), [4, 5])
        self.assertTrue(f_alias(data, "no_key") is None)
        self.assertEqual(f_alias(data, "no_key", return_all=True), [])
        self.assertTrue(f_alias(data, "energy") is None)

        file_path = f"{self.tmp_dir}/by_key.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        self.assertEqual(f_alias(file_path, "x_pos"), [1, 2, 3])
        self.assertEqual(f_alias(Path(file_path), "y_pos"), [4, 5])

        self.assertRaises(TypeError, f_alias, data, 1)
        self.assertRaises(TypeError, f_alias, data, "x_pos", return_all=1)
        self.assertRaises(TypeError, f_alias, 10, "x_pos")

    def test_load_csv(self):
        file_path = f"{self.tmp_dir}/load.csv"
        with open(file_path, "w") as f:
            f.write("1,2,3\n4,5,6\n")
        data = losa.load_csv(file_path)
        self.assertTrue(np.array_equal(data, [[1, 2, 3], [4, 5, 6]]))

        file_path = f"{self.tmp_dir}/load_semicolon.csv"
        with open(file_path, "w") as f:
            f.write("1;2\n3;4\n")
        data = losa.load_csv(file_path, dtype=np.int32, delimiter=";")
        self.assertTrue(data.dtype == np.int32)
        self.assertTrue(np.array_equal(data, [[1, 2], [3, 4]]))

        self.assertRaises(ValueError, losa.load_csv,
                          f"{self.tmp_dir}/csv_not_exist.csv")

    def test_save_csv(self):
        file_path = f"{self.tmp_dir}/save_csv/data.csv"
        data_in = np.random.rand(5, 3)
        losa.save_csv(file_path, data_in)
        self.assertTrue(os.path.isfile(file_path))
        self.assertTrue(np.allclose(losa.load_csv(file_path), data_in))

        losa.save_csv(file_path, [[1, 2], [3, 4]])
        self.assertTrue(np.array_equal(losa.load_csv(file_path),
                                       [[1, 2], [3, 4]]))

        self.assertRaises(ValueError, losa.save_csv, file_path, "abc")
        self.assertRaises(ValueError, losa.save_csv, file_path, (1, 2))

    def test_save_image_plt(self):
        mat = np.random.rand(32, 64)
        file_path = f"{self.tmp_dir}/plt/image.png"
        path = losa.save_image_plt(file_path, mat, dpi=50, fig_size=(4, 4))
        self.assertTrue(isinstance(path, str) and os.path.isfile(path))

        path = losa.save_image_plt(f"{self.tmp_dir}/plt/image2.png", mat,
                                   x_max=10.0, y_max=5.0, cmap="gray",
                                   color_bar=False, dpi=50, fig_size=(4, 4))
        self.assertTrue(os.path.isfile(path))

    def test_save_plot(self):
        y_list = np.random.rand(50)
        file_path = f"{self.tmp_dir}/plt/plot.png"
        path = losa.save_plot(file_path, y_list, dpi=50, fig_size=(4, 4))
        self.assertTrue(isinstance(path, str) and os.path.isfile(path))

        path = losa.save_plot(f"{self.tmp_dir}/plt/plot2.png", y_list,
                              x_list=np.arange(50), x_lim=(0, 40),
                              y_lim=(0, 1), dpi=50, fig_size=(4, 4))
        self.assertTrue(os.path.isfile(path))

    def test_save_q_plot(self):
        q_list = np.linspace(1.0, 5.0, 100)
        list_data = 1.0 + np.random.rand(100)
        file_path = f"{self.tmp_dir}/plt/q_plot.png"
        path = losa.save_q_plot(file_path, q_list, list_data, dpi=50,
                                fig_size=(6, 4))
        self.assertTrue(isinstance(path, str) and os.path.isfile(path))

        path = losa.save_q_plot(f"{self.tmp_dir}/plt/q_plot2.png", q_list,
                                list_data, q_standard=[1.5, 2.5, 3.5],
                                start=10, stop=90, x_ticks=[1, 2, 3, 4],
                                y_scale="linear", y_lim=(0, 3), dpi=50,
                                fig_size=(6, 4))
        self.assertTrue(os.path.isfile(path))
