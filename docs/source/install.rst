============
Installation
============

Algodif is a Python library, not an app. Users write Python scripts, or use
the provided GUIs/CLIs, to process their data. For beginners, a quick way to
get started with Python programming is to install
`Miniconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`__
or `Anaconda <https://www.anaconda.com/download>`__, then use an IDE such as
Spyder, PyCharm, or Visual Studio Code to write and run Python codes. It's a
good practice to create a separate Python environment for Algodif. Instructions
of how to create a new environment and how to install new packages are
`here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__
and `here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html>`__.
Note that the IDE needs to be configured to point to the new environment.

Using conda
-----------

- Install Miniconda as instructed above.
- Open a terminal or command prompt and run the following commands:

    + If installing to an existing environment:

      .. code-block:: console

         conda install -c conda-forge algodif

    + If installing to a new environment:

      .. code-block:: console

         conda create -n algodif
         conda activate algodif
         conda install python
         conda install -c conda-forge algodif

Using pip
---------

- Install Miniconda as instructed above.
- Open a terminal or command prompt and run the following commands:

    + If installing to an existing environment:

      .. code-block:: console

         pip install algodif

    + If installing to a new environment:

      .. code-block:: console

         conda create -n algodif
         conda activate algodif
         conda install python
         pip install algodif

Installing from source
----------------------

- Download the source from `github <https://github.com/algodif/algodif>`__
  (click -> Code -> Download ZIP) and unzip to a local folder, or clone the
  repository:

  .. code-block:: console

     git clone https://github.com/algodif/algodif.git

- Open a terminal or command prompt, navigate to the source folder, and run
  the following commands:

  .. code-block:: console

     conda create -n algodif
     conda activate algodif
     conda install python
     pip install .

- Use the package:

  .. code-block:: python

     import algodif.io.loadersaver as losa
     import algodif.prep.preprocessing as prep
     import algodif.proc.processing as proc
     import algodif.post.postprocessing as post
     import algodif.util.utility as util

Dependencies for GUIs/CLIs
--------------------------

The GUIs in the `GUIs_CLIs <https://github.com/algodif/algodif/tree/main/GUIs_CLIs>`__
folder are built with Tkinter and Matplotlib. The 4D data viewer additionally
requires PyVista, PyVistaQt, and PySide6:

.. code-block:: console

   conda install -c conda-forge pyvista pyvistaqt pyside6

The scripts can be organized and launched through the
`ScriptRunner GUI <https://pypi.org/project/scriptrunner-gui/>`__:

.. code-block:: console

   pip install scriptrunner-gui
