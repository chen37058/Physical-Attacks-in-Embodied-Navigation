"""
This script is intended for rendering a movie using the mitsuba renderer
"""

import mitsuba as mi
import drjit as dr
import csv
import os

if __name__ == "__main__":
    mi.set_variant("scalar_rgb")
    scene_path = os.path.join("scenes", "intersection_taxi")

    # read in a list of values used for animation
    # specify the key of the object to animate
    # specify sensor (camera) to use
    animations_path = os.path.join(scene_path,"animations", "cam_moves.csv")
    render_path = os.path.join("renders/stop_sign_approach_2")
    data = csv.reader(open(animations_path))
    # invert values for blender/mitsuba compatability
    moves = [-float(d[0]) for d in data]
    animateable_param = 'PerspectiveCamera_10.to_world'
    sensor = 10

    scene = mi.load_file(os.path.join(scene_path,"intersection_taxi.xml"))
    params = mi.traverse(scene)
    params.keep(animateable_param)
    for i in range(0, len(moves)):
        fn = str(i).rjust(4, '0')
        fn = f'{fn}.png'
        cam_position = params[animateable_param]
        mat = cam_position.matrix
        mat[3][2] = moves[i]
        params.update()
        print(f'rendering frame {i}')
        img = mi.render(scene, spp=256, sensor=sensor)
        mi.util.write_bitmap(os.path.join(render_path, fn), img)
