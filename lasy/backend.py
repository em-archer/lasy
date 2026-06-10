import os
from copy import deepcopy
from contextlib import contextmanager

lasy_backend = "AUTO"
if "LASY_BACKEND" in os.environ:
    assert os.environ["LASY_BACKEND"] in ["NP", "CP", "TORCH", "AUTO"], (
        "The enviroment variable 'LASY_BACKEND' must be one of "
        + "'NP' (NumPy), 'CP' (CuPy), 'TORCH' (PyTorch) or 'AUTO'!"
    )
    lasy_backend = os.environ["LASY_BACKEND"]

if lasy_backend == "AUTO":
    try:
        import torch as xp

        xp.set_default_dtype(xp.float64)  # Ensures numpy consistent precision

        # xp.is_available() might cause a CUDARuntimeError
        lasy_backend = "TORCH" if xp.cuda.is_available() else "NP"

        if lasy_backend == "TORCH":
            print("Using torch backend.")
            from scipy.interpolate import RegularGridInterpolator
            from scipy.signal import hilbert, zoom_fft
            from scipy.special import j0

            def to_cpu(arr):
                """Convert array from torch to numpy."""
                if isinstance(arr, xp.Tensor):
                    return arr.detach().cpu().numpy()
                elif isinstance(arr, list):
                    return [to_cpu(a) for a in arr]
                elif isinstance(arr, tuple):
                    return tuple(to_cpu(a) for a in arr)
                else:
                    return arr

            def to_gpu(arr):
                """Convert array from numpy to torch."""
                if not isinstance(arr, xp.Tensor):
                    import numpy as _np

                    if isinstance(arr, _np.ndarray):
                        return xp.from_numpy(arr)
                    else:
                        return xp.tensor(arr)
                else:
                    return arr
        else:
            print("Using numpy backend.")
            import numpy as xp
            from scipy.interpolate import RegularGridInterpolator
            from scipy.signal import hilbert, zoom_fft
            from scipy.special import j0

            def to_cpu(arr):
                """Convert array to numpy (no-op for numpy backend)."""
                return arr

            def to_gpu(arr):
                """Convert array to numpy (no-op for numpy backend)."""
                return arr

    except (ImportError, ModuleNotFoundError):
        try:
            import cupy as xp
            from cupyx.scipy.interpolate import RegularGridInterpolator
            from cupyx.scipy.signal import hilbert, zoom_fft
            from cupyx.scipy.special import j0

            # xp.is_available() might cause a CUDARuntimeError
            lasy_backend = "CP" if xp.is_available() else "NP"

            if lasy_backend == "CP":
                print("Using cupy backend.")

                def to_cpu(arr):
                    """Convert array from cupy to numpy."""
                    if isinstance(arr, xp.ndarray):
                        return xp.asnumpy(arr)
                    elif isinstance(arr, list):
                        return [to_cpu(a) for a in arr]
                    elif isinstance(arr, tuple):
                        return tuple(to_cpu(a) for a in arr)
                    else:
                        return arr

                def to_gpu(arr):
                    """Convert array from numpy to cupy."""
                    if not isinstance(arr, xp.ndarray):
                        return xp.asarray(arr)
                    else:
                        return arr

        except (ImportError, ModuleNotFoundError):
            print("Using numpy backend.")
            lasy_backend = "NP"
            import numpy as xp
            from scipy.interpolate import RegularGridInterpolator
            from scipy.signal import hilbert, zoom_fft
            from scipy.special import j0

            def to_cpu(arr):
                """Convert array to numpy (no-op for numpy backend)."""
                return arr

            def to_gpu(arr):
                """Convert array to numpy (no-op for numpy backend)."""
                return arr


def as_array(arr, dtype=None):
    """
    Convert input to appropriate array type based on active backend.

    - PyTorch backend: converts to torch.Tensor
    - CuPy backend: converts to cupy array
    - NumPy backend: converts to numpy array

    Parameters
    ----------
    arr : array-like
        Input array or data
    dtype : data-type, optional
        Data type of output array

    Returns
    -------
    array
        Array in the appropriate backend format
    """
    if lasy_backend == "TORCH":
        import numpy as _np

        if isinstance(arr, xp.Tensor):
            if dtype is not None:
                return arr.to(dtype=_get_torch_dtype(dtype))
            return arr
        else:
            if isinstance(arr, _np.ndarray):
                tensor = xp.from_numpy(arr)
            else:
                tensor = xp.tensor(_np.asarray(arr))
            if dtype is not None:
                return tensor.to(dtype=_get_torch_dtype(dtype))
            return tensor
    elif lasy_backend == "CP":
        if isinstance(arr, xp.ndarray):
            if dtype is not None:
                return arr.astype(dtype)
            return arr
        else:
            return xp.asarray(arr, dtype=dtype)
    else:
        if isinstance(arr, xp.ndarray):
            if dtype is not None:
                return arr.astype(dtype)
            return arr
        else:
            return xp.asarray(arr, dtype=dtype)


def copy(item, preserve_grad=True):
    """
    Copy an item.

    Parameters
    ----------
    item : tensor, array-like, list, tuple, dict
        The item to copy.
    preserve_grad : bool
        If True, the copy stays attached to the graph (gradients
        flow through it) and if False (default), the copy is detached.

    Returns
    -------
    A copy of arr with the same type and device as the input.
    """
    from copy import deepcopy
    if lasy_backend == "TORCH":
        import numpy as _np
    
        if isinstance(item, xp.Tensor):
            return item.clone() if preserve_grad else item.detach().clone()
    
        elif isinstance(item, _np.ndarray):
            return item.copy()
    
        elif isinstance(item, (list, tuple)):
            cloned = [copy(v, preserve_grad) for v in item]
            return type(item)(cloned)
    
        elif isinstance(item, dict):
            return {k: copy(v, preserve_grad) for k, v in item.items()}
    
        elif hasattr(item, '__dict__'):
            return _copy_object(item, preserve_grad)
    
        else:
            return deepcopy(item)
    else:
        preserve_grad=False
        if isinstance(item, xp.ndarray):
            return item.copy()
    
        elif isinstance(item, (list, tuple)):
            cloned = [copy(v, preserve_grad) for v in item]
            return type(item)(cloned)
    
        elif isinstance(item, dict):
            return {k: copy(v, preserve_grad) for k, v in item.items()}
    
        elif hasattr(item, '__dict__'):
            return _copy_object(item, preserve_grad)
    
        else:
            return deepcopy(item)


def _copy_object(obj, preserve_grad):
    """
    Helper function for duplication of instances of classes, e.g. the grid

    Parameters
    ----------
    obj : object
        The class instance to copy.
    preserve_grad : bool
        If True, the copy stays attached to the graph (gradients
        flow through it) and if False (default), the copy is detached.

    Returns
    -------
    A copy of obj with the same type and device as the input.
    """
    from copy import copy as _copy
    new_obj = _copy(obj)
    for attr, val in obj.__dict__.items():
        object.__setattr__(new_obj, attr, copy(val, preserve_grad))
    return new_obj


def _get_torch_dtype(dtype):
    """
    Convert numpy dtype or string dtype to torch dtype.

    Parameters
    ----------
    dtype : numpy.dtype, type, str, or torch.dtype
        Input dtype to convert

    Returns
    -------
    torch.dtype
        Corresponding torch dtype
    """
    import numpy as np
    import torch

    dtype_map = {
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.float16: torch.float16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.int16: torch.int16,
        np.int8: torch.int8,
        np.uint8: torch.uint8,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128,
    }

    # Handle string dtypes
    string_dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "int32": torch.int32,
        "int64": torch.int64,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }

    if isinstance(dtype, str):
        return string_dtype_map.get(dtype, torch.float32)

    if isinstance(dtype, type):
        dtype = np.dtype(dtype)

    if isinstance(dtype, np.dtype):
        return dtype_map.get(dtype.type, torch.float32)

    return dtype


def get_dtype(dtype_str):
    """
    Get the appropriate dtype for the active backend from a string.

    For torch backend, converts to torch dtype.
    For cupy and numpy backends, returns the string directly.

    Parameters
    ----------
    dtype_str : str
        Dtype string (e.g., 'complex128', 'float64')

    Returns
    -------
    dtype
        Backend-appropriate dtype object or string
    """
    if lasy_backend == "TORCH":
        return _get_torch_dtype(dtype_str)
    else:
        return dtype_str


def unwrap(p, discont=None, axis=-1, *, period=2 * __import__("math").pi):
    """
    Unwrap a phase-like tensor along the given axis for the torch backend.

    This mirrors the behavior of :func:`numpy.unwrap` / :func:`cupy.unwrap`, and
    is implemented to work with PyTorch tensors when `lasy_backend` is set to "TORCH".
    For "NP" and "CP" backends, it delegates to the native implementation.

    Parameters
    ----------
    p : array-like
        Input array or tensor.
    discont : float, optional, default=None
        Maximum discontinuity between values
    axis : int, optional, default=-1
        Axis along which to unwrap
    period : float, optional, default=2*pi
        Size of the range over which the input wraps

    Returns
    -------
    Tensor or ndarray
        Unwrapped array in the same backend as the input.
    """
    # If using torch backend, implement unwrap using torch operations
    if lasy_backend == "TORCH":
        p = as_array(p)

        nd = p.dim()
        # Use torch.diff (xp.diff) to compute adjacent differences
        try:
            dd = xp.diff(p, dim=axis)
        except TypeError:
            # Fallback if older torch versions use different signature
            dd = p.diff(dim=axis)

        if discont is None:
            discont = period / 2

        # Prepare slices to assign the unwrapped values
        slice1 = [slice(None)] * nd
        slice1[axis] = slice(1, None)
        slice1 = tuple(slice1)

        interval_high = period / 2
        interval_low = -interval_high

        # Compute modulo with correct centering in [-interval_high, interval_high]
        ddmod = xp.remainder(dd - interval_low, period) + interval_low
        # Boundary handling (ambiguous case)
        ddmod = xp.where((ddmod == interval_low) & (dd > 0), interval_high, ddmod)

        ph_correct = ddmod - dd
        ph_correct = xp.where(
            xp.abs(dd) < discont, xp.zeros_like(ph_correct), ph_correct
        )

        up = p.clone()
        up[slice1] = p[slice1] + xp.cumsum(ph_correct, dim=axis)
        return up
    else:
        # For numpy/cupy backends, delegate to their native implementation
        return xp.unwrap(p, discont=discont, axis=axis, period=period)


def gradient(input, spacing=1, axis=None, edge_order=1):
    """
    Compute the gradient of an n-dimensional array for the torch backend.

    This mirrors the behavior of :func:`numpy.gradient` / :func:`cupy.gradient`, and
    is implemented to work with PyTorch tensors when `lasy_backend` is set to "TORCH".
    For "NP" and "CP" backends, it delegates to the native implementation.

    Parameters
    ----------
    input : array-like
        Input array or tensor.
    spacing : scalar or array-like, optional
        Spacing between sample points. Default is 1.
    dim : int or sequence of ints, optional
        Dimensions along which the gradient is computed. Default is None (all dimensions).
    edge_order : int, optional
        Order of the finite difference approximation used at the boundaries. Default is 1.

    Returns
    -------
    list of arrays or tensors
        Gradient along each specified dimension in the same backend as the input.
    """

    # If using torch backend, implement gradient using torch operations
    if lasy_backend == "TORCH":
        input = as_array(input)
        if axis is not None:
            dim = (axis,)
        try:
            spacing = float(spacing)  # Convert to float
        except ValueError:
            spacing = (spacing,)  # Or ensure tuple for torch gradient
        return xp.gradient(input, spacing=spacing, dim=dim, edge_order=edge_order)[0]
    else:
        return xp.gradient(input, spacing, axis=axis, edge_order=edge_order)


def interp(x, x_points, y_points):
    """
    One-dimensional linear interpolation for the torch backend.

    This mirrors the behavior of :func:`numpy.interp` / :func:`cupy.interp`, and
    is implemented to work with PyTorch tensors when `lasy_backend` is set to "TORCH".
    For "NP" and "CP" backends, it delegates to the native implementation.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        x_points: the :math:`x`-coordinates of the data points, must be increasing.
        y_points: the :math:`y`-coordinates of the data points, same length as `x_points`.

    Returns
    -------
        the interpolated values, same size as `x`.
    """
    x = as_array(x)
    x_points = as_array(x_points)
    y_points = as_array(y_points)

    if lasy_backend == "TORCH":
        import numpy as _np

        return as_array(
            _np.interp(to_cpu(x), to_cpu(x_points), to_cpu(y_points)),
            dtype=y_points.dtype,
        )
    else:
        return xp.interp(x, x_points, y_points)


def weighted_avg(data, weights):
    """Compute the weighted average of data.

    Parameters
    ----------
    data : ndarray
        The data to average.
    weights : ndarray
        The weights for each data point.

    Returns
    -------
    float
        The weighted average of the data.
    """
    return xp.sum(data * weights) / xp.sum(weights)


@contextmanager
def no_grad():
    """Context manager that applies torch.no_grad() only when using the torch backend."""
    if lasy_backend == "TORCH":
        import torch
        with torch.no_grad():
            yield
    else:
        yield
        

__all__ = [
    "xp",
    "lasy_backend",
    "RegularGridInterpolator",
    "hilbert",
    "zoom_fft",
    "j0",
    "to_cpu",
    "to_gpu",
    "as_array",
    "copy",
    "get_dtype",
    "unwrap",
    "gradient",
    "interp",
    "weighted_avg",
]

print(f"Active backend: {lasy_backend}")
