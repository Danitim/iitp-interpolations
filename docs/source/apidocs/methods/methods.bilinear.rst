:py:mod:`methods.bilinear`
==========================

.. py:module:: methods.bilinear

.. autodoc2-docstring:: methods.bilinear
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`bilinear_interpolation <methods.bilinear.bilinear_interpolation>`
     - .. autodoc2-docstring:: methods.bilinear.bilinear_interpolation
          :summary:
   * - :py:obj:`_bilinear_gray <methods.bilinear._bilinear_gray>`
     - .. autodoc2-docstring:: methods.bilinear._bilinear_gray
          :summary:

API
~~~

.. py:function:: bilinear_interpolation(image: numpy.ndarray, new_height: int, new_width: int) -> numpy.ndarray
   :canonical: methods.bilinear.bilinear_interpolation

   .. autodoc2-docstring:: methods.bilinear.bilinear_interpolation

.. py:function:: _bilinear_gray(image: numpy.ndarray, new_h: int, new_w: int, channel: int) -> numpy.ndarray
   :canonical: methods.bilinear._bilinear_gray

   .. autodoc2-docstring:: methods.bilinear._bilinear_gray
