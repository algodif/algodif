.. _usage:

=====
Usage
=====

Resources
---------

-   A paper describing the methods developed and implemented in Algodif is
    under review :cite:`Vo:2026`.
-   Examples of how to use the package are in the "/examples" folder at the
    `github page <https://github.com/algodif/algodif/tree/main/examples>`__.
    Data used in the examples are in the
    `data <https://github.com/algodif/algodif/tree/main/data>`__ folder.
-   GUIs/CLIs for processing and visualizing EDDT data are in the
    `GUIs_CLIs <https://github.com/algodif/algodif/tree/main/GUIs_CLIs>`__ folder.

Notes on calibration coefficients
---------------------------------

Calibration methods return polynomial coefficients ordered with the constant
term first, i.e. :math:`E(x) = a_0 + a_1 x + a_2 x^2 + ...` where :math:`x` is
the pixel index. The same convention is used for q-calibration. For a 2D image,
the coefficients are returned as a 2D array with one row per detector strip,
which can be saved and reused by the post-processing methods to correct
other datasets acquired under the same conditions.

Demonstrations
--------------

.. toctree::

    usage/demo_01
    usage/demo_02
    usage/demo_03
    usage/demo_04
    usage/demo_05
    usage/demo_06
    usage/demo_07

User interface software
-----------------------

.. toctree::

    usage/gui_cli
