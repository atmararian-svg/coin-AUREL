"""
AUREL coin - GLB verification + studio preview renders.

  blender --background --python scripts/render_previews.py [-- --quick]
  (or: python scripts/render_previews.py [--quick])

Imports aurel_coin_final.glb into a FRESH empty scene (so only what survived the
export is rendered), prints an inventory, then renders under a temporary soft
studio light rig (nothing here is saved into the GLB).

glTF -> Blender import: the coin's front (+Z in glTF) faces -Y, monogram up = +Z.
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Euler, Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLB = os.path.join(ROOT, "aurel_coin_final.glb")
PREV = os.path.join(ROOT, "previews")
os.makedirs(PREV, exist_ok=True)
QUICK = "--quick" in sys.argv
RES = 560 if QUICK else 1100
SAMPLES = 24 if QUICK else 128


def inventory():
    print("=" * 60)
    print("GLB:", GLB, f"{os.path.getsize(GLB) / 1e6:.2f} MB")
    print("objects:", [(o.name, o.type) for o in bpy.data.objects])
    print("cameras:", len(bpy.data.cameras), "lights:", len(bpy.data.lights))
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    assert len(meshes) == 1, "expected exactly one mesh object"
    ob = meshes[0]
    me = ob.data
    tris = sum(len(p.vertices) - 2 for p in me.polygons)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    w = ob.matrix_world
    print(f"mesh '{ob.name}': {len(me.vertices):,} verts, {tris:,} triangles")
    print("location", tuple(ob.location), "rotation", tuple(ob.rotation_euler), "scale", tuple(ob.scale))
    print("bbox min", co.min(0).round(5), "max", co.max(0).round(5), "centre", ((co.min(0) + co.max(0)) / 2).round(6))
    dims = co.max(0) - co.min(0)
    print(f"diameter {dims[0]:.4f}  thickness {dims[1]:.4f}  ratio {dims[0] / dims[1]:.1f}:1")
    print("uv layers:", [u.name for u in me.uv_layers])
    for m in me.materials:
        print("material:", m.name)
        for n in m.node_tree.nodes:
            if n.type == "TEX_IMAGE":
                img = n.image
                print(f"   image {img.name} {img.size[0]}x{img.size[1]} -> "
                      f"{[l.to_socket.name + '@' + l.to_node.name for l in n.outputs['Color'].links]}")
            if n.type == "BSDF_PRINCIPLED":
                print("   principled metallic linked:", n.inputs["Metallic"].is_linked,
                      "roughness linked:", n.inputs["Roughness"].is_linked,
                      "normal linked:", n.inputs["Normal"].is_linked)
    print("images:", [(i.name, tuple(i.size)) for i in bpy.data.images])
    print("=" * 60)
    return ob


def studio(scene):
    world = bpy.data.worlds.new("studio")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    # environment seen in reflections: soft vertical gradient (bright top, dark floor)
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[0].color = (0.010, 0.010, 0.011, 1)
    ramp.color_ramp.elements[1].position = 0.85
    ramp.color_ramp.elements[1].color = (0.55, 0.53, 0.50, 1)
    tc2 = nt.nodes.new("ShaderNodeTexCoord")
    sep2 = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc2.outputs["Generated"], sep2.inputs[0])
    nt.links.new(sep2.outputs["Z"], ramp.inputs["Fac"])
    env = nt.nodes.new("ShaderNodeBackground")
    env.inputs["Strength"].default_value = 0.6
    nt.links.new(ramp.outputs["Color"], env.inputs["Color"])
    cam_bg = nt.nodes.new("ShaderNodeBackground")
    cam_bg.inputs["Color"].default_value = (0.018, 0.018, 0.02, 1)
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(env.outputs["Background"], mix.inputs[1])
    nt.links.new(cam_bg.outputs["Background"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    def area(name, loc, size, energy, color=(1, 1, 1)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.shape = "DISK"
        ld.size = size
        ld.energy = energy
        ld.color = color
        o = bpy.data.objects.new(name, ld)
        scene.collection.objects.link(o)
        o.location = loc
        d = Vector((0, 0, 0)) - Vector(loc)
        o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        return o

    area("key", (-4.0, -5.0, 4.5), 5.0, 1400)
    area("fill", (5.5, -3.5, 1.0), 6.0, 450)
    area("top", (0.5, 1.0, 7.0), 6.0, 600)
    area("rim", (2.5, 6.0, 3.0), 4.0, 700)
    area("low", (-3.0, -2.0, -5.0), 5.0, 160)


def camera(scene, loc, lens=105):
    cd = bpy.data.cameras.new("cam")
    cd.lens = lens
    cd.clip_start = 0.05
    o = bpy.data.objects.new("cam", cd)
    scene.collection.objects.link(o)
    o.location = loc
    d = Vector((0, 0, 0)) - Vector(loc)
    o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    scene.camera = o
    return o


def render(scene, path, res=RES):
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("rendered", path)


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=GLB)
    coin = inventory()
    coin.rotation_mode = "XYZ"          # glTF importer uses quaternions; we spin with Euler
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Base Contrast"
    studio(scene)

    D = 7.2
    views = {
        "front": ((0, -D, 0), (0, 0, 0)),
        "back": ((0, D, 0), (0, 0, 0)),
        "edge": ((D, 0, 0), (0, 0, 0)),
        "perspective": (None, (0, 0, 0)),
        "near_edge": (None, (0, 0, 0)),
    }
    cam = camera(scene, (0, -D, 0))
    for name in views:
        coin.rotation_euler = (0, 0, 0)
        if name == "front":
            cam.location = (0, -D, 0)
        elif name == "back":
            cam.location = (0, D, 0)
        elif name == "edge":
            cam.location = (D, 0, 0)
        elif name == "perspective":       # 40 deg turn + slight elevation
            a = math.radians(40)
            cam.location = (D * math.sin(a), -D * math.cos(a), 0.9)
        elif name == "near_edge":          # 80 deg: almost edge-on
            a = math.radians(80)
            cam.location = (D * math.sin(a), -D * math.cos(a), 0.6)
        d = Vector((0, 0, 0)) - cam.location
        cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        render(scene, os.path.join(PREV, f"aurel_{name}.png"))

    # edge close-up so the engraving can actually be judged
    a = math.radians(84)
    cam.location = (2.0 * math.sin(a), -2.0 * math.cos(a), 0.15)
    d = Vector((0.98, 0, 0.0)) - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = 70
    render(scene, os.path.join(PREV, "aurel_edge_closeup.png"))

    # rotation under fixed light: spin about the vertical axis
    cam.data.lens = 105
    cam.location = (0, -D, 0.5)
    d = Vector((0, 0, 0)) - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    frames = []
    for i, deg in enumerate([0, 30, 60, 85, 120, 150, 180, 210]):
        coin.rotation_euler = Euler((0, 0, math.radians(deg)))
        p = os.path.join(PREV, f"_spin_{i}.png")
        render(scene, p, res=360 if not QUICK else 240)
        frames.append(p)
    imgs = [bpy.data.images.load(p) for p in frames]
    w, h = imgs[0].size
    sheet = np.zeros((h * 2, w * 4, 4), np.float32)
    for i, im in enumerate(imgs):
        px = np.array(im.pixels[:], np.float32).reshape(h, w, 4)
        r, c = 1 - i // 4, i % 4
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = px
    out = bpy.data.images.new("spin", w * 4, h * 2)
    out.pixels = sheet.ravel()
    out.filepath_raw = os.path.join(PREV, "aurel_rotation_sheet.png")
    out.file_format = "PNG"
    out.save()
    for p in frames:
        os.remove(p)
    print("rotation sheet written")


main()
