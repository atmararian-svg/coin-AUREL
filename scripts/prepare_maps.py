"""
AUREL coin - reference -> relief/material map preparation.

Turns the three AUREL reference photos into clean, lighting-independent
data that the Blender build script consumes:

  build/front_height.npy   front face height field (coin units, rim top = 0)
  build/back_height.npy    back face height field
  build/edge_height.npy    one seamless period of the edge ornament (radial offset)
  build/atlas_basecolor.png / atlas_orm.png   2048x2048 PBR atlas
  build/debug_*.png        inspection images

Coordinate conventions (shared with build_aurel_coin.py):
  coin radius R = 1.0, thickness T = 0.1  (diameter : thickness = 20 : 1)
  face maps are S x S images covering x,y in [-1,1]; row 0 = +y (top)
  atlas: front face -> top-left 1024^2, back face -> top-right 1024^2,
         edge strip  -> bottom 2048 x 1024 (one ornament period across the width)

Run with any Python that has numpy, scipy, opencv, scikit-image and Pillow:
  python scripts/prepare_maps.py
"""
import os

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as nd
from skimage.filters import sato

import coin_spec as spec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "reference")
OUT = os.path.join(ROOT, "build")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- constants
S = 1024                 # face map resolution (matches reference resolution)
ATLAS = 2048
BAND_HALF = spec.BAND_HALF

# depth levels (coin units, relative to the outer rim top = 0)
FIELD = -0.0065          # base field
MONOGRAM = 0.0058        # front monogram height above field
MOTIF = 0.0052           # back central motif height above field
STAR = 0.0040            # back four-point stars
BAND_ORN = 0.0040        # border ornament height above field
EDGE_ORN = 0.0024        # edge ornament height above band floor

# fitted ellipse of the coin silhouette in each reference: cx, cy, w, h, angle
ELLIPSE = {
    "front": (650.41, 582.04, 1042.05, 1051.22, 0.1077),
    "back": (622.65, 588.09, 1061.30, 1066.22, 118.46),
}


def ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def band(r, a, b, soft):
    return ss(a - soft, a + soft, r) * (1.0 - ss(b - soft, b + soft, r))


def norm01(x, lo=1, hi=99, mask=None):
    v = x[mask] if mask is not None else x
    a, b = np.percentile(v, [lo, hi])
    return np.clip((x - a) / (b - a + 1e-9), 0, 1)


def save_gray(path, x):
    Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8)).save(path)


def unwarp(name, size=S):
    """Resample the photographed coin into a square image whose unit circle is the coin edge."""
    cx, cy, w, h, a = ELLIPSE[name]
    im = np.asarray(Image.open(os.path.join(REF, f"aurel_{name}.webp")).convert("RGB")).astype(np.float32)
    R = size / 2
    yy, xx = np.mgrid[0:size, 0:size]
    u = (xx + 0.5 - R) / R
    v = (yy + 0.5 - R) / R
    t = np.deg2rad(a)
    ca, sa = np.cos(t), np.sin(t)
    pu = u * ca + v * sa
    pv = -u * sa + v * ca
    X = cx + (pu * w / 2) * ca - (pv * h / 2) * sa
    Y = cy + (pu * w / 2) * sa + (pv * h / 2) * ca
    return cv2.remap(im, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REPLICATE)


def luminance(rgb):
    lab = cv2.cvtColor(np.clip(rgb / 255.0, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)
    return lab[..., 0] / 100.0


def ridge_mask(L, valid):
    """Soft 'raised stroke' mask from multi-scale ridge detection (lighting independent-ish)."""
    s1 = sato(L, sigmas=(2, 3, 4), black_ridges=False)
    s2 = sato(L, sigmas=(5, 7, 9), black_ridges=False)
    s1 /= np.percentile(s1[valid], 99)
    s2 /= np.percentile(s2[valid], 99)
    m = np.maximum(ss(0.18, 0.55, s1), ss(0.22, 0.62, s2))
    m = nd.grey_closing(m, size=3)
    return nd.gaussian_filter(m, 1.0)


def clean_mask(m, thr, min_area, hole_area, close=2):
    M = m > thr
    if close:
        M = nd.binary_closing(M, structure=np.ones((3, 3)), iterations=close)
    # fill small holes only
    filled = nd.binary_fill_holes(M)
    holes, n = nd.label(filled & ~M)
    if n:
        areas = nd.sum(np.ones_like(holes), holes, index=np.arange(1, n + 1))
        small = np.isin(holes, np.nonzero(areas < hole_area)[0] + 1)
        M = M | small
    lab, n = nd.label(M)
    if n:
        areas = nd.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        M = np.isin(lab, np.nonzero(areas >= min_area)[0] + 1)
    return M


def drop_outer_arcs(M, r, r_max):
    """Remove components that sit on the inner ring (arcs of the frame mis-detected as relief)."""
    lab, n = nd.label(M)
    if not n:
        return M
    idx = np.arange(1, n + 1)
    mean_r = nd.mean(r, lab, idx)
    return np.isin(lab, idx[mean_r < r_max])


def smooth_mask(M, sigma):
    return nd.gaussian_filter(M.astype(np.float32), sigma) > 0.5


def dome(M, width):
    """Rounded embossed cross-section from a binary mask."""
    d = nd.distance_transform_edt(M)
    t = np.clip(d / width, 0, 1)
    return nd.gaussian_filter(np.sin(t * np.pi / 2) ** 0.75, 0.7)


def ring(r, c, w, h):
    """Rounded (half-cosine) raised ring."""
    t = np.clip(np.abs(r - c) / (w / 2), 0, 1)
    return h * (0.5 + 0.5 * np.cos(np.pi * t)) ** 0.6


def radial_profile(r):
    """Machined circular structure shared by both faces (clean, symmetric)."""
    P = np.full_like(r, FIELD)
    # inner double ring framing the central field
    P += ring(r, 0.684, 0.020, 0.0030) + ring(r, 0.709, 0.014, 0.0024)
    P -= ring(r, 0.669, 0.006, 0.0007) + ring(r, 0.723, 0.006, 0.0007)
    # double ring between ornament band and bead channel
    P += ring(r, 0.913, 0.016, 0.0030) + ring(r, 0.929, 0.008, 0.0020)
    # bead channel slightly sunk
    P -= 0.0005 * band(r, 0.935, 0.963, 0.002)
    # outer rim: flat top at 0 with a rounded inner shoulder
    rim = ss(0.959, 0.972, r)
    P = P * (1 - rim) + rim * (-0.0002 * (1 - ss(0.972, 0.985, r)))
    return P


def beads(xn, yn, r):
    th = np.arctan2(yn, xn)
    out = np.zeros_like(r)
    for n_off, rad, h in ((0.0, 0.0108, 0.0050), (0.5, 0.0042, 0.0024)):
        k = np.round(th / (2 * np.pi) * 168 - n_off) + n_off
        tk = k * 2 * np.pi / 168
        bx, by = 0.9485 * np.cos(tk), 0.9485 * np.sin(tk)
        d = np.hypot(xn - bx, yn - by) / rad
        out = np.maximum(out, h * np.sqrt(np.clip(1 - d * d, 0, 1)) ** 0.9)
    return out


def scratches(shape, n, seed, length=(0.02, 0.15), px=1.0):
    """Fine random hairline scratches (value 0..1)."""
    rng = np.random.default_rng(seed)
    hgt, wid = shape
    img = np.zeros(shape, np.float32)
    for _ in range(n):
        x0, y0 = rng.uniform(0, wid), rng.uniform(0, hgt)
        ang = rng.uniform(0, np.pi)
        ln = rng.uniform(*length) * wid
        x1, y1 = x0 + np.cos(ang) * ln, y0 + np.sin(ang) * ln
        cv2.line(img, (int(x0 * 16), int(y0 * 16)), (int(x1 * 16), int(y1 * 16)),
                 float(rng.uniform(0.3, 1.0)), 1, cv2.LINE_AA, shift=4)
    return nd.gaussian_filter(img, 0.35 * px)


# ---------------------------------------------------------------- faces
def build_face(name):
    rgb = unwarp(name)
    L = luminance(rgb)
    yy, xx = np.mgrid[0:S, 0:S]
    xn = (xx + 0.5 - S / 2) / (S / 2)
    yn = -(yy + 0.5 - S / 2) / (S / 2)
    r = np.hypot(xn, yn)
    valid = r < 0.93

    m = ridge_mask(L, valid)
    hp_fine = L - nd.gaussian_filter(L, 2.5)      # engraving / hammer detail
    hp_mid = L - nd.gaussian_filter(L, 8.0)       # internal modelling of ornaments

    zone_field = 1 - ss(0.640, 0.660, r)
    zone_band = band(r, 0.738, 0.902, 0.008)

    # --- central element (monogram / floral motif): highest level
    Mc = clean_mask(m * (r < 0.655), 0.40, 140, 260) & (r < 0.655)
    Mc = smooth_mask(drop_outer_arcs(Mc, r, 0.56), 1.3)
    Mc = nd.binary_dilation(Mc, iterations=1) & (r < 0.655)   # reference strokes are bolder than the ridge mask
    dc = dome(Mc, 8.0)
    Mc_s = nd.gaussian_filter(Mc.astype(np.float32), 1.0)
    amp = MONOGRAM if name == "front" else MOTIF
    central = amp * (0.70 * dc + 0.30 * m * Mc_s) + 0.0006 * hp_mid * Mc_s

    if name == "back":
        # four-point stars: small, crisp, lower than the motif
        stars = np.zeros_like(r)
        for sx in (-0.545, 0.545):
            dx, dy = np.abs(xn - sx), np.abs(yn - 0.0)
            # astroid-like four point star
            q = (dx / 0.080) ** 0.62 + (dy / 0.105) ** 0.62
            stars = np.maximum(stars, STAR * np.clip(1 - q, 0, 1) ** 0.7)
        star_zone = np.zeros_like(r, bool)
        for sx in (-0.545, 0.545):
            star_zone |= np.hypot(xn - sx, yn) < 0.11
        central = np.where(star_zone, stars, central)
        Mc = np.where(star_zone, stars > 1e-5, Mc)

    # --- border ornament band: primary + secondary levels from ridge strength
    Mb = smooth_mask(clean_mask(m * zone_band, 0.36, 40, 80, close=1), 0.9)
    db = dome(Mb, 5.0)
    bandrel = BAND_ORN * (0.55 * db + 0.45 * m) * zone_band + 0.0005 * hp_mid * zone_band

    # --- base field texture: hammered / aged surface (recessed micro detail)
    field_mask = np.clip(zone_field * (1 - Mc_s) + zone_band * (1 - nd.gaussian_filter(Mb.astype(np.float32), 1.0)), 0, 1)
    tex = 0.00045 * np.clip(hp_fine, -0.25, 0.25) * field_mask

    scr = scratches((S, S), 260, seed=11 if name == "front" else 23, px=1.0)

    H = radial_profile(r) + beads(xn, yn, r)
    H = H + central * zone_field + bandrel + tex - 0.00006 * scr
    H = np.minimum(H, -0.0002 * (1 - ss(0.959, 0.972, r)))  # relief never proud of the rim

    # material helpers (all in face-image space)
    relief = H - FIELD
    cavity = nd.gaussian_filter(H, 6) - H          # >0 in recesses
    cav = norm01(cavity, 2, 99.5, r < 0.99)
    raised = np.clip(relief / 0.005, 0, 1)
    np.save(os.path.join(OUT, f"{name}_height.npy"), H.astype(np.float32))

    save_gray(os.path.join(OUT, f"debug_{name}_mask.png"), np.maximum(Mc * 1.0, Mb * 0.6))
    save_gray(os.path.join(OUT, f"debug_{name}_height.png"), (H - H[r < 0.99].min()) / (0 - H[r < 0.99].min()))
    return dict(H=H, cav=cav, raised=raised, scr=scr, r=r, tex=hp_fine * field_mask)


# ---------------------------------------------------------------- edge
def build_edge_tile():
    """Extract one stretch of the edge vine, undo the cylinder foreshortening,
    mirror it into a seamless period and convert to a radial displacement."""
    a = np.asarray(Image.open(os.path.join(REF, "aurel_edge.webp")).convert("RGB")).astype(np.float32)
    L = luminance(a)
    yc, Rpx = 512.0, 452.0
    x0, x1 = 750.0, 790.0                        # band interior between its grooves
    y_flower = 548.0                             # mirror line through the central flower
    s0 = Rpx * np.arcsin((y_flower - yc) / Rpx)
    seg_len = 235.0                              # arc-length pixels in one half period
    n_s, n_z = 512, 96
    s = s0 + np.linspace(0, seg_len, n_s)
    ys = yc + Rpx * np.sin(s / Rpx)
    xs = np.linspace(x0, x1, n_z)
    X, Y = np.meshgrid(xs, ys)                   # rows: along circumference, cols: across band
    seg = cv2.remap(L.astype(np.float32), X.astype(np.float32), Y.astype(np.float32), cv2.INTER_CUBIC)
    tile = np.concatenate([seg, seg[::-1][1:-1]], axis=0).T   # (n_z, period) z across rows
    # remove lighting gradient across the band, find raised ornament
    Lt = tile - nd.gaussian_filter(tile, (6, 18), mode="wrap") + 0.5
    s1 = sato(Lt, sigmas=(1.5, 2.5, 3.5), black_ridges=False)
    s1 /= np.percentile(s1, 99)
    bright = ss(0.47, 0.60, nd.gaussian_filter(Lt, 1.0, mode="wrap"))
    m = np.clip(0.55 * ss(0.15, 0.6, s1) + 0.6 * bright, 0, 1)
    m = nd.gaussian_filter(m, 0.8, mode="wrap")
    M = m > 0.45
    d = nd.distance_transform_edt(np.pad(M, ((0, 0), (40, 40)), mode="wrap"))[:, 40:-40]
    dm = np.sin(np.clip(d / 3.0, 0, 1) * np.pi / 2) ** 0.75
    E = EDGE_ORN * (0.6 * dm + 0.4 * m) + 0.0004 * (Lt - 0.5)
    # fade ornament out at the band borders (grooves)
    zz = np.linspace(-1, 1, tile.shape[0])[:, None]
    E *= 1 - ss(0.88, 0.99, np.abs(zz))
    E = np.maximum(E, 0)
    np.save(os.path.join(OUT, "edge_height.npy"), E.astype(np.float32))
    save_gray(os.path.join(OUT, "debug_edge_tile.png"), E / E.max())
    save_gray(os.path.join(OUT, "debug_edge_src.png"), norm01(tile))
    return E


# ---------------------------------------------------------------- atlas
GOLD_HI = np.array([0.84, 0.66, 0.36])    # polished raised metal (linear-ish sRGB values below)
GOLD = np.array([0.76, 0.58, 0.30])       # ~#C2944C muted warm gold
GOLD_LO = np.array([0.55, 0.41, 0.21])    # recess
PATINA = np.array([0.33, 0.25, 0.13])     # very subtle oxidation in deepest recesses


def face_material(f):
    cav, raised, scr = f["cav"], f["raised"], f["scr"]
    rng = np.random.default_rng(5)
    blot = nd.gaussian_filter(rng.standard_normal((S, S)), 22)
    blot = norm01(blot)
    fld = (1 - raised)[..., None]
    col = GOLD[None, None] * (1 - raised[..., None] * 0.6) + GOLD_HI[None, None] * raised[..., None] * 0.6
    col *= 1 - 0.12 * fld                                   # field slightly duller than polished relief
    col = col * (1 - 0.30 * cav[..., None]) + GOLD_LO[None, None] * 0.30 * cav[..., None]
    pat = np.clip((cav - 0.55) / 0.45, 0, 1) * (0.35 + 0.4 * blot)
    col = col * (1 - 0.45 * pat[..., None]) + PATINA[None, None] * 0.45 * pat[..., None]
    col *= (0.96 + 0.06 * blot)[..., None]
    col *= (1 + 0.10 * np.clip(f["tex"], -0.2, 0.2) * 3)[..., None]
    rough = 0.33 - 0.11 * raised + 0.07 * cav + 0.05 * (blot - 0.5) + 0.07 * scr
    ao = 1 - 0.35 * cav ** 1.5
    return np.clip(col, 0, 1), np.clip(rough, 0.18, 0.5), np.clip(ao, 0, 1)


def edge_material(E):
    """Material for the side strip (atlas bottom half: 2048 along circumference x 1024 across profile)."""
    W, Hh = ATLAS, ATLAS // 2
    pts, arc, _ = spec.side_profile(256)
    zs = pts[:, 1]
    total = arc[-1]
    p = (np.arange(Hh) + 0.5) / Hh * total                   # arc position for each strip row
    z = np.interp(p, arc, zs)
    # ornament displacement resampled into the strip
    in_band = np.abs(z) < BAND_HALF
    zi = (BAND_HALF - z) / (2 * BAND_HALF) * (E.shape[0] - 1)  # tile row index (row 0 = +z)
    ui = (np.arange(W) + 0.5) / W * E.shape[1]
    U, Zi = np.meshgrid(ui, np.clip(zi, 0, E.shape[0] - 1))
    Es = cv2.remap(np.concatenate([E, E[:, :8]], axis=1).astype(np.float32),
                   U.astype(np.float32), Zi.astype(np.float32), cv2.INTER_CUBIC)
    Es = Es * in_band[:, None]
    raised = np.clip(Es / EDGE_ORN, 0, 1)
    groove = np.exp(-((np.abs(z) - BAND_HALF) / 0.0015) ** 2)[:, None] * np.ones((1, W))
    band_floor = (in_band[:, None] * (1 - raised))
    cav = np.clip(0.75 * band_floor + 0.8 * groove, 0, 1)
    rng = np.random.default_rng(9)
    blot = norm01(nd.gaussian_filter(rng.standard_normal((Hh, W)), (20, 40), mode="wrap"))
    scr = scratches((Hh, W), 220, seed=31, length=(0.01, 0.06))
    col = GOLD[None, None] * (1 - raised[..., None] * 0.35) + GOLD_HI[None, None] * raised[..., None] * 0.35
    col = col * (1 - 0.32 * cav[..., None]) + GOLD_LO[None, None] * 0.32 * cav[..., None]
    pat = np.clip((cav - 0.5) / 0.5, 0, 1) * (0.3 + 0.4 * blot)
    col = col * (1 - 0.4 * pat[..., None]) + PATINA[None, None] * 0.4 * pat[..., None]
    col *= (0.96 + 0.06 * blot)[..., None]
    rough = 0.27 - 0.05 * raised + 0.09 * cav + 0.04 * (blot - 0.5) + 0.06 * scr
    ao = 1 - 0.35 * cav ** 1.5
    return np.clip(col, 0, 1), np.clip(rough, 0.18, 0.5), np.clip(ao, 0, 1)


def to_srgb8(c):
    return (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    front = build_face("front")
    back = build_face("back")
    E = build_edge_tile()

    base = np.zeros((ATLAS, ATLAS, 3), np.float32)
    orm = np.zeros((ATLAS, ATLAS, 3), np.float32)
    orm[..., 0], orm[..., 1], orm[..., 2] = 1.0, 0.3, 1.0
    base[:] = GOLD

    # both face images are stored as seen by the viewer; the UV mapping
    # (coin_spec.uv_back) takes care of the back being viewed from -Z
    for f, x0 in ((front, 0), (back, S)):
        col, rough, ao = face_material(f)
        base[0:S, x0:x0 + S] = col
        orm[0:S, x0:x0 + S, 0] = ao
        orm[0:S, x0:x0 + S, 1] = rough

    col, rough, ao = edge_material(E)
    base[S:, :] = col
    orm[S:, :, 0] = ao
    orm[S:, :, 1] = rough

    Image.fromarray(to_srgb8(base)).save(os.path.join(OUT, "atlas_basecolor.png"))
    Image.fromarray(to_srgb8(orm)).save(os.path.join(OUT, "atlas_orm.png"))
    print("maps written to", OUT)


if __name__ == "__main__":
    main()
