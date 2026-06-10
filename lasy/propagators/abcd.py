from lasy.backend import as_array, xp, no_grad


class ABCD:
    r"""
    Class that defines and manipulates ABCD ray matrices for an optical system.

    Parameters
    ----------
    abcd : 2d array
        The 2D ray matrix of the optical system through which the beam propagates.
        By default, this is initialised to be the unitary matrix:


        .. math::

            O =
            \begin{pmatrix}
            1 & 0 \\
            0 & 1
            \end{pmatrix}.

    """

    def __init__(self, abcd=None):
        super().__init__()
        with no_grad():
            if abcd is None:
                abcd = as_array([[1.0, 0.0], [0.0, 1.0]])
            self.abcd = abcd

    def add_vacuum(self, distance):
        r"""
        Add a propagation over a distance :math:`z` in vacuum.

        Parameters
        ----------
        distance : float (in meter)
            The distance in free-space which the beam propagates.
            The ray matrix for propagation of a distance :math:`z` in vacuum is:

            .. math::

                O =
                \begin{pmatrix}
                1 & z \\
                0 & 1
                \end{pmatrix}.

        """
        with no_grad():
            vacuum = as_array([[1.0, float(distance)], [0.0, 1.0]])
            self.abcd = xp.matmul(vacuum, self.abcd)

    def add_lens(self, focal_length):
        r"""
        Add a thin-lens with a focal length :math:`f_0`.

        Parameters
        ----------
        focal_length : float (in meter)
            The focal length of a thin-lens through which the beam propagates.
            The ray matrix for propagation through a thin lens with focal length :math:`f_0` is:

            .. math::

                O =
                \begin{pmatrix}
                1 & 0 \\
                -1/f_0 & 1
                \end{pmatrix}.

        """
        with no_grad():
            lens = as_array([[1.0, 0.0], [-1.0 / focal_length, 1.0]])
            self.abcd = xp.matmul(lens, self.abcd)
