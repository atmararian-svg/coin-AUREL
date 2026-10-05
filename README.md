# AUREL coin – web GLB

A 3D model of the AUREL coin for use with WebGL / Three.js. The front monogram, the back floral motif, the borders, the beads and the edge vine are real raised relief. A baked normal map adds the fine engraving and the hammered texture on top.

| File | |
|---|---|
| `aurel_coin_final.glb` | final asset (glTF 2.0 binary, textures embedded) |
| `aurel_coin.blend` | Blender 4.2 source (decimated web mesh and material, images packed) |
| `scripts/build_aurel_coin.py` | Blender build: geometry, decimation, normal bake, material, export |
| `scripts/prepare_maps.py` | turns the reference images into height fields and the PBR atlas |
| `scripts/coin_spec.py` | shared dimensions, edge profile and UV layout |
| `scripts/render_previews.py` | re-imports the GLB into an empty scene, prints an inventory, renders studio previews |
| `previews/` | front, back, edge, edge close-up, 40° perspective, 80° near-edge, rotation sheet |
| `reference/` | the supplied AUREL front, back and edge references |

## Asset facts

* 180,000 triangles and 90,950 vertices in one mesh, one material and one primitive.
* Diameter 2.0 and thickness 0.1, so diameter : thickness = 20 : 1. The origin is at the exact centre and transforms are applied.
* Orientation in glTF: the front (monogram) faces **+Z** and the monogram's up direction is **+Y**. The coin faces a default Three.js camera with no rotation needed.
* Material `AUREL_Gold` is a glTF metallic-roughness material:
  * base colour, a muted warm gold close to `#C2944C`;
  * ORM texture: occlusion, roughness about 0.22 on polished relief to 0.40 in recesses, metallic 1.0;
  * tangent-space normal map, with tangents exported (MikkTSpace).
* All three textures are 2048² JPEG. UV atlas layout: front face top-left, back face top-right, one edge-ornament period across the bottom half. The edge pattern repeats 9 times using `REPEAT` wrapping, so there is no seam.
* The material is single-sided (`doubleSided: false`) because the mesh is a closed solid.
* The GLB contains no cameras, lights, animations, reference planes or unused data.

Relief depths, relative to the rim top: field −0.0065; monogram up to +0.0058 above the field; back motif +0.0052; border ornament +0.0040; stars +0.0040; edge vine +0.0024 above the sunken edge band.

## Rebuilding

```bash
pip install bpy==4.2.0 numpy scipy opencv-python-headless scikit-image pillow
python scripts/prepare_maps.py            # about 10 s  -> build/*.npy, build/atlas_*.png
python scripts/build_aurel_coin.py        # about 75 s  -> aurel_coin_final.glb, aurel_coin.blend
python scripts/render_previews.py         # verification and previews (--quick for a fast pass)
```

With a normal Blender install, `blender --background --python scripts/build_aurel_coin.py` also works. `prepare_maps.py` needs the extra Python packages listed above, so run it with a system Python.

## Using it in Three.js

```js
const gltf = await new GLTFLoader().loadAsync('aurel_coin_final.glb');
const coin = gltf.scene.getObjectByName('AUREL_Coin');
scene.add(gltf.scene);
// Metals need something to reflect: use an environment map, e.g. RoomEnvironment
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment()).texture;
```

The file is about 7.9 MB, and most of that is uncompressed vertex data. To make it smaller for production, run
`npx @gltf-transform/cli meshopt aurel_coin_final.glb aurel_coin_web.glb`, or use `draco` instead of `meshopt`. You then need to register `MeshoptDecoder` (or `DRACOLoader`) with the `GLTFLoader`.
