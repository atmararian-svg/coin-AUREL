"""
AUREL coin - Blender build script.

  blender --background --python scripts/build_aurel_coin.py
  (or, with the `bpy` pip module:  python scripts/build_aurel_coin.py)

Requires the maps produced by scripts/prepare_maps.py in build/.

Pipeline
  1. HIGH  mesh: ~6M triangles, every face/edge texel displaced (bake source only)
  2. MID   mesh: same construction at lower density, then quadric-decimated to the web budget
  3. bake  tangent-space normal map HIGH -> LOW (Cycles, selected-to-active)
  4. one antique-gold Principled material: basecolor + ORM + normal (glTF compatible)
  5. orient for three.js (front face -> +Z, monogram upright -> +Y), apply transforms
  6. export aurel_coin_final.glb, save aurel_coin.blend

The coin is ONE closed manifold built from concentric rings:
  front centre -> front rings -> rim bevel -> edge band -> rim bevel -> back rings -> back centre
"""
import math
import os
import sys
import time

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coin_spec as spec  # noqa: E402

ROOT = os.path.dirname(HERE)
BUILD = os.path.join(ROOT, "build")
OUT_GLB = os.path.join(ROOT, "aurel_coin_final.glb")
OUT_BLEND = os.path.join(ROOT, "aurel_coin.blend")

TARGET_TRIS = 180_000
HIGH = dict(face_dr=1 / 512, side_n=spec.EDGE_PERIODS * 1024, band_rows=144)
MID = dict(face_dr=1 / 300, side_n=spec.EDGE_PERIODS * 288, band_rows=56)

t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- maps
H_FRONT = np.load(os.path.join(BUILD, "front_height.npy"))
H_BACK = np.load(os.path.join(BUILD, "back_height.npy"))
E_TILE = np.load(os.path.join(BUILD, "edge_height.npy"))


def bilinear(img, row, col, wrap_cols=False):
    h, w = img.shape
    r0 = np.floor(row).astype(int)
    c0 = np.floor(col).astype(int)
    fr, fc = row - r0, col - c0
    r0c, r1c = np.clip(r0, 0, h - 1), np.clip(r0 + 1, 0, h - 1)
    if wrap_cols:
        c0c, c1c = c0 % w, (c0 + 1) % w
    else:
        c0c, c1c = np.clip(c0, 0, w - 1), np.clip(c0 + 1, 0, w - 1)
    return ((img[r0c, c0c] * (1 - fc) + img[r0c, c1c] * fc) * (1 - fr)
            + (img[r1c, c0c] * (1 - fc) + img[r1c, c1c] * fc) * fr)


def face_height(img, x, y):
    """img is stored as seen by the viewer, covering [-1,1]^2, row 0 = +y."""
    s = img.shape[0]
    return bilinear(img, (1 - y) / 2 * s - 0.5, (x + 1) / 2 * s - 0.5)


def edge_offset(theta, z):
    rows, period = E_TILE.shape
    col = (theta * spec.EDGE_PERIODS / (2 * np.pi)) % 1.0 * period
    row = (spec.BAND_HALF - z) / (2 * spec.BAND_HALF) * (rows - 1)
    return bilinear(E_TILE, np.clip(row, 0, rows - 1), col - 0.5, wrap_cols=True)


# ---------------------------------------------------------------- ring mesh
def zipper(na, nb):
    """Triangulate the strip between ring A (na verts) and ring B (nb verts).
    Returns (ia, ib, kind) per triangle, unwrapped indices (may equal na / nb)."""
    fa = np.arange(1, na + 1) / na
    fb = np.arange(1, nb + 1) / nb
    ev = np.concatenate([fa, fb])
    typ = np.concatenate([np.zeros(na, int), np.ones(nb, int)])
    order = np.lexsort((typ, ev))
    typ = typ[order]
    i = np.concatenate([[0], np.cumsum(typ == 0)[:-1]])
    j = np.concatenate([[0], np.cumsum(typ == 1)[:-1]])
    return i, j, typ


def build_coin(cfg):
    """Return verts (N,3), tris (M,3), corner uvs (M,3,2)."""
    dr = cfg["face_dr"]
    pts, arc, is_band = spec.side_profile(cfg["band_rows"])
    arc_total = arc[-1]

    # ring descriptors: (kind, param) ; kind in front/back (param=radius) or side (param=profile idx)
    n_face_rings = int(round(spec.FACE_R / dr))
    radii = np.linspace(0, spec.FACE_R, n_face_rings + 1)
    rings = [("front", rr) for rr in radii]
    rings += [("side", k) for k in range(1, len(pts) - 1)]
    rings += [("back", rr) for rr in radii[::-1]]

    def ring_count(kind, p):
        if kind == "side":
            return cfg["side_n"]
        if p == 0:
            return 1
        return max(6, int(round(2 * np.pi * p / dr)))

    verts, ring_off, ring_n = [], [], []
    off = 0
    for kind, p in rings:
        n = ring_count(kind, p)
        th = 2 * np.pi * np.arange(n) / n
        if kind == "side":
            r0, z = pts[p]
            rr = np.full(n, r0)
            if is_band[p]:
                rr = rr + edge_offset(th, np.full(n, z))
            v = np.stack([rr * np.cos(th), rr * np.sin(th), np.full(n, z)], 1)
        else:
            x, y = p * np.cos(th), p * np.sin(th)
            if kind == "front":
                z = spec.HALF_T + face_height(H_FRONT, x, y)
            else:  # back is viewed from -Z: viewer's x = -world x
                z = -(spec.HALF_T + face_height(H_BACK, -x, y))
            v = np.stack([x, y, z], 1)
        verts.append(v)
        ring_off.append(off)
        ring_n.append(n)
        off += n
    verts = np.concatenate(verts).astype(np.float64)

    n_front = len(radii)                    # rings[0 .. n_front-1] are front
    tris, uvs = [], []
    for k in range(len(rings) - 1):
        na, nb = ring_n[k], ring_n[k + 1]
        ia, ib, typ = zipper(na, nb)
        # corners: (ring, unwrapped index)
        c0 = np.stack([np.full_like(ia, k), ia], 1)
        c1 = np.stack([np.full_like(ib, k + 1), ib], 1)
        c2 = np.where(typ[:, None] == 0, np.stack([np.full_like(ia, k), ia + 1], 1),
                      np.stack([np.full_like(ib, k + 1), ib + 1], 1))
        corners = np.stack([c0, c1, c2], 1)              # (m,3,2)
        rk = corners[..., 0]
        idx = corners[..., 1]
        n_of = np.where(rk == k, na, nb)
        vid = np.where(rk == k, ring_off[k], ring_off[k + 1]) + idx % n_of
        # drop degenerate triangles at the centre vertex
        good = ~((vid[:, 0] == vid[:, 1]) | (vid[:, 1] == vid[:, 2]) | (vid[:, 0] == vid[:, 2]))
        vid, idx, n_of = vid[good], idx[good], n_of[good]
        theta = 2 * np.pi * idx / n_of                     # unwrapped angle per corner
        P = verts[vid]
        if k < n_front - 1:
            u, v = spec.uv_front(P[..., 0], P[..., 1])
        elif k >= len(rings) - n_front:
            u, v = spec.uv_back(P[..., 0], P[..., 1])
        else:
            # side band k spans profile points k-(n_front-1) .. +1
            pk = k - (n_front - 1)
            rk_g = np.where(corners[good][..., 0] == k, pk, pk + 1)
            u, v = spec.uv_side(theta, arc[rk_g], arc_total)
        tris.append(vid)
        uvs.append(np.stack([u, v], -1))
    return verts, np.concatenate(tris), np.concatenate(uvs)


def make_object(name, verts, tris, uvs):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    me.loops.add(tris.size)
    me.loops.foreach_set("vertex_index", tris.astype(np.int32).ravel())
    me.polygons.add(len(tris))
    me.polygons.foreach_set("loop_start", np.arange(0, tris.size, 3, dtype=np.int32))
    uvl = me.uv_layers.new(name="UVMap")
    uvl.data.foreach_set("uv", uvs.astype(np.float32).ravel())
    me.update(calc_edges=True)
    me.validate(clean_customdata=False)
    me.polygons.foreach_set("use_smooth", np.ones(len(tris), bool))
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


# ---------------------------------------------------------------- material
def load_image(path, colorspace):
    img = bpy.data.images.load(path, check_existing=True)
    img.colorspace_settings.name = colorspace
    return img


def gltf_output_group():
    """Custom group recognised by the glTF exporter for the occlusion texture."""
    name = "glTF Material Output"
    if name in bpy.data.node_groups:
        return bpy.data.node_groups[name]
    g = bpy.data.node_groups.new(name, "ShaderNodeTree")
    g.interface.new_socket("Occlusion", in_out="INPUT", socket_type="NodeSocketFloat")
    g.nodes.new("NodeGroupInput")
    return g


def make_material(normal_img):
    mat = bpy.data.materials.new("AUREL_Gold")
    mat.use_nodes = True
    mat.use_backface_culling = True      # closed solid -> glTF doubleSided: false
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (-300, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.28

    tex = lambda img, y: (lambda n: (setattr(n, "image", img), setattr(n, "location", (-900, y)), n)[-1])(
        nt.nodes.new("ShaderNodeTexImage"))
    base = tex(load_image(os.path.join(BUILD, "atlas_basecolor.png"), "sRGB"), 300)
    orm = tex(load_image(os.path.join(BUILD, "atlas_orm.png"), "Non-Color"), 0)
    nrm = tex(normal_img, -300)
    nt.links.new(base.outputs["Color"], bsdf.inputs["Base Color"])
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    sep.location = (-600, 0)
    nt.links.new(orm.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
    nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nmap.location = (-600, -300)
    nt.links.new(nrm.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    grp = nt.nodes.new("ShaderNodeGroup")
    grp.node_tree = gltf_output_group()
    grp.location = (-300, -500)
    nt.links.new(sep.outputs["Red"], grp.inputs["Occlusion"])
    return mat, nrm


# ---------------------------------------------------------------- main
def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    log("building HIGH mesh")
    v, f, uv = build_coin(HIGH)
    high = make_object("AUREL_high", v, f, uv)
    log(f"HIGH: {len(v):,} verts {len(f):,} tris")

    log("building MID mesh")
    v, f, uv = build_coin(MID)
    low = make_object("AUREL_Coin", v, f, uv)
    log(f"MID: {len(v):,} verts {len(f):,} tris")

    # merge coincident ring seams (keeps per-loop UVs), then decimate to the web budget
    bpy.context.view_layer.objects.active = low
    for o in scene.objects:
        o.select_set(o is low)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-7)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    dec = low.modifiers.new("Decimate", "DECIMATE")
    dec.decimate_type = "COLLAPSE"
    dec.ratio = TARGET_TRIS / len(low.data.polygons)
    dec.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=dec.name)
    tri = low.modifiers.new("Triangulate", "TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=tri.name)
    log(f"LOW: {len(low.data.vertices):,} verts {len(low.data.polygons):,} tris")

    # ---- bake normal map high -> low
    normal_img = bpy.data.images.new("AUREL_normal", spec.ATLAS, spec.ATLAS, alpha=False, float_buffer=True)
    normal_img.colorspace_settings.name = "Non-Color"
    normal_img.generated_color = (0.5, 0.5, 1.0, 1.0)
    mat, nrm_node = make_material(normal_img)
    low.data.materials.append(mat)
    nt = mat.node_tree
    for n in nt.nodes:
        n.select = False
    nt.nodes.active = nrm_node
    # high poly needs a material slot too (unused by the bake)
    high.data.materials.append(mat)

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 4
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.cage_extrusion = 0.004
    scene.render.bake.max_ray_distance = 0.010
    scene.render.bake.margin = 12
    scene.render.bake.normal_space = "TANGENT"
    for o in scene.objects:
        o.select_set(o in (high, low))
    bpy.context.view_layer.objects.active = low
    log("baking normal map ...")
    # disconnect the (still blank) normal map while baking so the bake is not fed by itself
    nmap = nt.nodes["Normal Map"]
    link = [l for l in nt.links if l.to_node == nmap][0]
    nt.links.remove(link)
    bpy.ops.object.bake(type="NORMAL")
    nt.links.new(nrm_node.outputs["Color"], nmap.inputs["Color"])
    log("bake done")
    normal_img.filepath_raw = os.path.join(BUILD, "atlas_normal.png")
    normal_img.file_format = "PNG"
    normal_img.save()
    # swap the bake buffer for a regular file image so it gets embedded like the others
    # (remove first: load(check_existing=True) would otherwise return the bake buffer itself)
    bpy.data.images.remove(normal_img)
    nrm_node.image = load_image(os.path.join(BUILD, "atlas_normal.png"), "Non-Color")

    # ---- cleanup: only the coin survives
    bpy.data.objects.remove(high, do_unlink=True)
    for m in list(bpy.data.meshes):
        if m.users == 0:
            bpy.data.meshes.remove(m)

    # ---- orient for three.js: front (+Z local) -> -Y Blender -> +Z glTF ; up (+Y) -> +Z Blender -> +Y glTF
    low.rotation_euler = (math.radians(90), 0, 0)
    for o in scene.objects:
        o.select_set(o is low)
    bpy.context.view_layer.objects.active = low
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # origin at the exact geometric centre of the coin's bounding box
    co = np.empty(len(low.data.vertices) * 3)
    low.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    centre = (co.min(0) + co.max(0)) / 2
    co -= centre
    low.data.vertices.foreach_set("co", co.ravel())
    low.data.update()
    log("bbox centre offset removed:", centre)

    # purge unused datablocks
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

    # ---- export
    kw = dict(filepath=OUT_GLB, export_format="GLB", use_selection=True, export_apply=True,
              export_yup=True, export_texcoords=True, export_normals=True, export_tangents=True,
              export_materials="EXPORT", export_cameras=False, export_lights=False,
              export_image_format="JPEG", export_animations=False)
    try:
        bpy.ops.export_scene.gltf(**kw, export_image_quality=92)
    except TypeError:
        bpy.ops.export_scene.gltf(**kw, export_jpeg_quality=92)
    log("exported", OUT_GLB, f"{os.path.getsize(OUT_GLB) / 1e6:.2f} MB")

    for img in bpy.data.images:
        if img.filepath:
            img.pack()
    bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND, compress=True)
    log("saved", OUT_BLEND)


main()
