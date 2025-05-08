:py:mod:`methods.lanczos`
=========================

.. py:module:: methods.lanczos

.. autodoc2-docstring:: methods.lanczos
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`sinc <methods.lanczos.sinc>`
     - .. autodoc2-docstring:: methods.lanczos.sinc
          :summary:
   * - :py:obj:`lanczos_kernel <methods.lanczos.lanczos_kernel>`
     - .. autodoc2-docstring:: methods.lanczos.lanczos_kernel
          :summary:
   * - :py:obj:`lanczos_interpolation <methods.lanczos.lanczos_interpolation>`
     - .. autodoc2-docstring:: methods.lanczos.lanczos_interpolation
          :summary:
   * - :py:obj:`_lanczos_gray <methods.lanczos._lanczos_gray>`
     - .. autodoc2-docstring:: methods.lanczos._lanczos_gray
          :summary:

API
~~~

.. py:function:: sinc(x: numpy.ndarray) -> numpy.ndarray
   :canonical: methods.lanczos.sinc

   .. autodoc2-docstring:: methods.lanczos.sinc

.. py:function:: lanczos_kernel(x: numpy.ndarray, a: int) -> numpy.ndarray
   :canonical: methods.lanczos.lanczos_kernel

   .. autodoc2-docstring:: methods.lanczos.lanczos_kernel

.. py:function:: lanczos_interpolation(image: numpy.ndarray, new_height: int, new_width: int, a: int = 3) -> numpy.ndarray
   :canonical: methods.lanczos.lanczos_interpolation

   .. autodoc2-docstring:: methods.lanczos.lanczos_interpolation

.. py:function:: _lanczos_gray(image: numpy.ndarray, new_h: int, new_w: int, a: int, channel: int | None = None) -> numpy.ndarray
   :canonical: methods.lanczos._lanczos_gray

   .. autodoc2-docstring:: methods.lanczos._lanczos_gray
