from lasy.backend import xp, lasy_backend, to_gpu, as_array, get_dtype
from scipy.constants import c

from lasy.optical_elements.optical_element import OpticalElement
from lasy.utils.refractive_index import Material


class PhasePlate(OpticalElement):
    r"""
    Class that represents an HOFI@MAGMA phase plate.

    Here the surface of the phase plate is defined by
    h(r) = h_0 + a_2 * r^2 + a_4 * r^4

    where we ignore the constant h_0 term.


    Parameters
    ----------
    a2: float (in meter)
        Coefficient from the equation above

    a4: float (in meter)
        Coefficient from the equation above

    material_name: string
        Name of material compatible with lasy.utils.refractive_index.Material.
    """

    def __init__(self, a2, a4, material_name="fused silica"):
        self.a2 = a2
        self.a4 = a4
        self.material = Material(name=material_name)

    def amplitude_multiplier(self, x, y, omega):
        """
        Return the amplitude multiplier.

        Parameters
        ----------
        x, y, omega : ndarrays of floats
            Define points on which to evaluate the multiplier.
            These arrays need to all have the same shape.

        Returns
        -------
        multiplier : ndarray of complex numbers
            Contains the value of the multiplier at the specified points.
            This array has the same shape as the array omega.
        """
        r2 = x**2 + y**2
        r = xp.sqrt(r2)

        k = omega / c
        wavelength = 2 * xp.pi / k
        wavelength_um = wavelength * 1e6  # Convert to micrometers

        eta = self.material.calc_n(as_array(wavelength_um))

        # Equation for optic surface
        h = self.a2 * r**2 + self.a4 * r**4

        # Calculate phase shift
        phi = k * (eta - 1) * h

        return xp.exp(1j * phi)


class VortexPlate(OpticalElement):
    r"""
    Class that represents Vortex phase plate.

    Such an optic adds an azimuthally varying phase which
    varies from 0 to 2*m*pi over the full circle.


    Parameters
    ----------
    m: float
        Coefficient from the equation above
    """

    def __init__(self, m):
        self.m = m

    def amplitude_multiplier(self, x, y, omega):
        """
        Return the amplitude multiplier.

        Parameters
        ----------
        x, y, omega : ndarrays of floats
            Define points on which to evaluate the multiplier.
            These arrays need to all have the same shape.

        Returns
        -------
        multiplier : ndarray of complex numbers
            Contains the value of the multiplier at the specified points.
            This array has the same shape as the array omega.
        """
        theta = xp.arctan2(y, x)

        return xp.exp(1j * self.m * theta)


class SpatialLightModulator(OpticalElement):
    r"""
    Class that represents a Spatial Light Modulator (SLM).

    Such an optic adds a phase defined by a 2D array.

    Parameters
    ----------
    phase_array: 2D ndarray of floats
        Array defining the phase shift at each pixel in radians.

    pixel_size: float (in meter)
        Size of each pixel in the phase_array.
    """

    def __init__(self, phase_array, pixel_size):
        self.phase_array = phase_array
        self.pixel_size = pixel_size
        self.nx, self.ny = phase_array.shape
        self.x_extent = self.nx * pixel_size / 2
        self.y_extent = self.ny * pixel_size / 2

    def amplitude_multiplier(self, x, y, omega):
        """
        Return the amplitude multiplier.

        Parameters
        ----------
        x, y, omega : ndarrays of floats
            Define points on which to evaluate the multiplier.
            These arrays need to all have the same shape.

        Returns
        -------
        multiplier : ndarray of complex numbers
            Contains the value of the multiplier at the specified points.
            This array has the same shape as the array omega.
        """
        phase_array = as_array(self.phase_array, dtype=get_dtype("complex128"))

        # Map x,y coordinates to pixel indices
        if lasy_backend == "TORCH":
            ix = xp.floor((x + self.x_extent) / self.pixel_size).to(get_dtype(int))
            iy = xp.floor((y + self.y_extent) / self.pixel_size).to(get_dtype(int))
        else:
            ix = xp.floor((x + self.x_extent) / self.pixel_size).astype(int)
            iy = xp.floor((y + self.y_extent) / self.pixel_size).astype(int)

        # Initialize phase array
        phase = xp.zeros_like(x, dtype=get_dtype("complex128"))

        # Apply phase where indices are within bounds
        valid = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        phase[valid] = phase_array[ix[valid], iy[valid]]

        # Create amplitude mask: 1 inside the SLM region, 0 outside
        amplitude_mask = xp.ones_like(x)
        amplitude_mask[~valid] = 0

        return amplitude_mask * xp.exp(1j * phase)
