"""
Shared AUREL coin specification (pure numpy - importable from Blender and plain Python).

Units: coin radius = 1.0 (diameter 2.0), thickness 0.1  ->  diameter : thickness = 20 : 1.
Front face looks toward +Z, back face toward -Z, origin at the exact centre.
"""
import numpy as np

R = 1.0
T = 0.1
HALF_T = T / 2
FACE_R = 0.990            # where the displaced face mesh ends and the side profile begins
EDGE_PERIODS = 9          # repeats of the ornament around the circumference
BAND_HALF = 0.031         # half height of the engraved band on the edge
BAND_FLOOR_R = 0.9965     # radius of the band floor (ornament rises from here)
CORNER_R = 0.007          # rounded rim bevel radius

FACE_MAP = 1024           # face height map resolution
ATLAS = 2048


def _arc(cx, cz, rad, a0, a1, n):
    a = np.linspace(a0, a1, n)
    return np.stack([cx + rad * np.cos(a), cz + rad * np.sin(a)], 1)


def side_profile(band_rows):
    """(r, z) polyline from the front rim edge, around the edge, to the back rim edge.

    Returns (pts[N,2], arc[N], is_band[N]).  `band_rows` controls vertical resolution of the band.
    """
    c = CORNER_R
    top = [np.array([[FACE_R, HALF_T], [1 - c, HALF_T]])]
    top.append(_arc(1 - c, HALF_T - c, c, np.pi / 2, 0.0, 10)[1:])           # rounded rim corner
    rail_lo = BAND_HALF + 0.0035
    top.append(np.array([[1.0, rail_lo + 0.0006],                             # rail
                         [0.9994, rail_lo],                                   # tiny chamfer
                         [0.9978, BAND_HALF + 0.0022],
                         [0.9958, BAND_HALF + 0.0010],                        # groove bottom
                         [0.9960, BAND_HALF + 0.0002]]))
    upper = np.concatenate(top, 0)
    zb = np.linspace(BAND_HALF, -BAND_HALF, band_rows)
    bandpts = np.stack([np.full_like(zb, BAND_FLOOR_R), zb], 1)
    lower = upper[::-1] * np.array([1, -1])
    pts = np.concatenate([upper, bandpts, lower], 0)
    is_band = np.zeros(len(pts), bool)
    is_band[len(upper):len(upper) + band_rows] = True
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    return pts, arc, is_band


def profile_arc_total():
    _, arc, _ = side_profile(64)
    return arc[-1]


# ---------------------------------------------------------------- UV atlas layout
def uv_front(x, y):
    """front face: top-left quadrant; image col grows with +x, row 0 = +y."""
    return 0.25 + 0.25 * x, 0.75 + 0.25 * y


def uv_back(x, y):
    """back face seen from -Z: viewer's right is world -X.  top-right quadrant."""
    return 0.75 - 0.25 * x, 0.75 + 0.25 * y


def uv_side(theta, arc_pos, arc_total):
    """side strip: bottom half, one ornament period across the full width (wraps with REPEAT)."""
    u = theta * EDGE_PERIODS / (2 * np.pi)
    v = 0.5 - 0.5 * arc_pos / arc_total
    return u, v
