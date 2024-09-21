import mitsuba as mi
import numpy as np
mi.set_variant('llvm_ad_rgb')
import mitsuba as mi
import numpy as np

scene = mi.load_file("scenes/00813-svBbv1Pavdk/00813-svBbv1Pavdk.xml")
params = mi.traverse(scene)
print(params)

'''
  adversarial_patch.bsdf.opacity.data                     ∂, D     TensorXf BitmapTexture
  adversarial_patch.bsdf.opacity.to_uv                    , D      ScalarTransform3f BitmapTexture
  adversarial_patch.bsdf.nested_bsdf.reflectance.data     ∂        TensorXf BitmapTexture
  adversarial_patch.bsdf.nested_bsdf.reflectance.to_uv             ScalarTransform3f BitmapTexture
  adversarial_patch.vertex_count                                   int   PLYMesh
  adversarial_patch.face_count                                     int   PLYMesh
  adversarial_patch.faces                                          UInt  PLYMesh
  adversarial_patch.vertex_positions                      ∂, D     Float PLYMesh
  adversarial_patch.vertex_normals                        ∂, D     Float PLYMesh
  adversarial_patch.vertex_texcoords                      ∂        Float PLYMesh 
'''