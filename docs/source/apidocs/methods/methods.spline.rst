:py:mod:`methods.spline`
========================

.. py:module:: methods.spline

.. autodoc2-docstring:: methods.spline
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`cubic_kernel <methods.spline.cubic_kernel>`
     - .. autodoc2-docstring:: methods.spline.cubic_kernel
          :summary:
   * - :py:obj:`_fast_bicubic_patch <methods.spline._fast_bicubic_patch>`
     - .. autodoc2-docstring:: methods.spline._fast_bicubic_patch
          :summary:
   * - :py:obj:`spline_interpolation <methods.spline.spline_interpolation>`
     - .. autodoc2-docstring:: methods.spline.spline_interpolation
          :summary:
   * - :py:obj:`_spline_gray <methods.spline._spline_gray>`
     - .. autodoc2-docstring:: methods.spline._spline_gray
          :summary:

API
~~~

.. py:function:: cubic_kernel(x: float) -> float
   :canonical: methods.spline.cubic_kernel

   .. autodoc2-docstring:: methods.spline.cubic_kernel

.. py:function:: _fast_bicubic_patch(patch: numpy.ndarray, dx: float, dy: float) -> float
   :canonical: methods.spline._fast_bicubic_patch

   .. autodoc2-docstring:: methods.spline._fast_bicubic_patch

.. py:function:: spline_interpolation(image: numpy.ndarray, new_height: int, new_width: int) -> numpy.ndarray
   :canonical: methods.spline.spline_interpolation

   .. autodoc2-docstring:: methods.spline.spline_interpolation

.. py:function:: _spline_gray(image: numpy.ndarray, new_h: int, new_w: int, channel: int | None = None) -> numpy.ndarray
   :canonical: methods.spline._spline_gray

   .. autodoc2-docstring:: methods.spline._spline_gray
