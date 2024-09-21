"""
Example Usage:python src/render_batch.py 
    -s scenes/cube_scene/cube_scene.xml 
    -cm generate_taxi_cam_positions
"""

import mitsuba as mi
import drjit as dr
import os
import argparse
import time
import ast

from dt2 import *

if __name__  == "__main__":
    parser = argparse.ArgumentParser( \
        description='Example script with default values' \
        ,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-s", "--scene", help="Mitsuba scene file path.", required=True)
    parser.add_argument("-sr", "--sensor-radius", type=float, help="sensor radius")
    parser.add_argument("-sc", "--sensor-count", type=int, help="sensor count")
    parser.add_argument("-sz", "--sensor-z-lats", type=ast.literal_eval, help="sensor z lats")
    parser.add_argument("-sp", "--spp", type=int, default=256, help="samples per pixel per render")
    parser.add_argument("-od", "--outdir", help="directory for rendered images", default="renders")
    parser.add_argument("-ck", "--cam-key", help="Mitsuba Scene Params Camera Key", default='PerspectiveCamera.to_world')

    args = parser.parse_args()

    sensor_z_lats = args.sensor_z_lats    
    
    mi.set_variant("cuda_ad_rgb")
    scene = mi.load_file(args.scene)
    
    camera_positions = generate_cam_positions_for_lats(sensor_z_lats \
                                                        ,args.sensor_radius \
                                                        , args.sensor_count)
    params = mi.traverse(scene)
    cam_key = args.cam_key
    print(f'rendering {len(camera_positions)} imgs...')
    for i in range(0, len(camera_positions)):
        params[cam_key].matrix = camera_positions[i].matrix
        params.update()
        img =  mi.render(scene, params=params, spp=256, sensor=0, seed=i+1)
        rendered_img_path = os.path.join(args.outdir,f"render_{i}.png")
        mi.util.write_bitmap(rendered_img_path, data=img)
        time.sleep(0.2)
    print('done')