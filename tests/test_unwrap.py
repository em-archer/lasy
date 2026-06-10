from lasy.backend import xp, as_array, unwrap


def test_unwrap_1d():
    import numpy as np

    p = np.array([0.0, np.pi / 2, np.pi, -3 * np.pi / 4])

    res = unwrap(as_array(p))
    expected = np.unwrap(p)

    assert xp.allclose(as_array(res), as_array(expected))


def test_unwrap_axis():
    import numpy as np

    p = np.array([[0.0, np.pi / 2, -3 * np.pi / 4], [0.1, 0.2, -3 * np.pi / 4 + 0.2]])

    res = unwrap(as_array(p), axis=-1)
    expected = np.unwrap(p, axis=-1)

    assert xp.allclose(as_array(res), as_array(expected))
