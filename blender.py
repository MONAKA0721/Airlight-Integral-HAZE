import bpy
import glob
import random
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import re
import mathutils
import math
import datetime
import os
import json

today_str = datetime.date.today().strftime('%m%d')
output_root = "/Volumes/WD_HDD_2TB/Dataset/"
output_filename = str(len(glob.glob(os.path.join(output_root, 'EXRfiles', today_str, '*.exr')))+1).zfill(4)

def format_mtl_file(layout_mtl_file):
    try:
        tmp_list = []
        firstSlash = layout_mtl_file.find("/")
        secondSlash = layout_mtl_file.find("/", firstSlash+1)
        scene = layout_mtl_file[firstSlash+1:secondSlash]
        with open(layout_mtl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line[:6] == "map_Kd":
                    index = line.find(" ") + 3
                    if line[-5:-1] == ".bmp":
                        end_index = line.rfind("/")
                        map_Kd = "./SceneNet" + line[index:end_index] + "/*.bmp"
                    else:
                        map_Kd = "./SceneNet" + line[index:-1] + "/*.bmp"

                    files = glob.glob(map_Kd)
                    file = random.choice(files)
                    new_line = file.replace("./SceneNet", "map_Kd ..")
                    tmp_list.append( new_line + "\n")
                else:
                    tmp_list.append(line)

        with open(layout_mtl_file, 'w', encoding="utf-8") as f:
            for line in tmp_list:
                f.write(line)

    except IOError:
        print ("Could not open the .mtl file...")

def render_with_blender(layout_file, lamp, camera_info):
    # bpy.ops.wm.open_mainfile(filepath="./untitled.blend")
    for item in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(item)

    for item in bpy.data.objects:
        bpy.data.objects.remove(item)

    for item in bpy.data.meshes:
        bpy.data.meshes.remove(item)

    for item in bpy.data.materials:
        bpy.data.materials.remove(item)

    # カメラを置く
    bpy.ops.object.camera_add()
    scene = bpy.context.scene # シーンの定義
    scene.camera = bpy.data.objects['Camera']
    scene.camera.location = camera_info["location"]
    scene.camera.rotation_mode = 'ZYX'
    scene.camera.rotation_euler = camera_info["euler"]
    bpy.data.cameras["Camera.001"].lens = 30

    # ライトを置く
    bpy.ops.object.light_add(type='POINT', location=lamp["location"])
    bpy.data.lights["Point"].energy = 300 # 単位はワット

    # レイアウトファイルのインポート
    bpy.ops.import_scene.obj(filepath=layout_file)

    #レンダリング
    # switch on nodes
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create input render layer node
    rl = tree.nodes.new('CompositorNodeRLayers')
    rl.location = 185,285

    # create output node
    v = tree.nodes.new('CompositorNodeComposite')
    v.location = 750,210
    v.use_alpha = False

    # Links
    links.new(rl.outputs[2], v.inputs[2])  # link Image output to Viewer input

    bpy.data.scenes[0].render.resolution_x = 640
    bpy.data.scenes[0].render.resolution_y = 480
    bpy.data.scenes[0].render.resolution_percentage = 100
    bpy.data.scenes[0].render.image_settings.file_format = 'OPEN_EXR'
    bpy.data.scenes[0].render.image_settings.use_zbuffer = True
    bpy.data.scenes[0].render.image_settings.color_depth = '32'
    bpy.ops.render.render(use_viewport=True)

    save_name = today_str + "/" + output_filename +  ".exr"
    os.makedirs(os.path.join(output_root, "Depth", today_str), exist_ok=True)
    bpy.data.images['Render Result'].save_render(filepath=os.path.join(output_root, "Depth", save_name))

    bpy.context.scene.use_nodes = False
    bpy.ops.render.render(use_viewport=True)

    os.makedirs(os.path.join(output_root, "EXRfiles", today_str), exist_ok=True)
    bpy.data.images['Render Result'].save_render(filepath=output_root + "EXRfiles/" + save_name)

def output_lamp_camera_information(camera, lamp, layout_obj_file):
    save_name = today_str + "/" + output_filename + '.txt'
    save_name = output_root + "camera_lamp_information/" + save_name
    s = "camera_location: " + str(camera["location"]) + "\n" + \
        "camera_euler(zyx): " + str(camera["euler"])  + "\n" + \
        "lamp_location: " + str(lamp["location"]) + "\n" + \
        "\n" + \
        "created_by: " + layout_obj_file

    os.makedirs(os.path.join(output_root, "camera_lamp_information", today_str), exist_ok=True)
    with open(save_name, mode='w') as f:
        f.write(s)

def calculate_camera_rotation(camera_location, attention_point):
    camera_location = np.array([camera_location[0], camera_location[1], camera_location[2]])

    v = np.array(attention_point) - camera_location
    r_3 = -v/np.linalg.norm(v)
    r_1 = - np.cross(r_3, np.array([0, 0, 1]))
    r_2 = np.cross(r_1, r_3)
    rotation_matrix = np.concatenate([r_1.reshape(3, -1), r_2.reshape(3, -1), r_3.reshape(3, -1)], axis=1)
    gl2bl = np.array([[1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]])
    rot = Rotation.from_matrix(rotation_matrix)
    r = rotation_matrix
    matrix = mathutils.Matrix(((r[0][0],r[0][1], r[0][2]), (r[1][0], r[1][1], r[1][2]), (r[2][0], r[2][1], r[2][2])))
    euler = matrix.to_euler('ZYX')
    return euler

def determine_lamp_and_camera(layout_obj_file):
    bathroom_x_min = [-0.6, -1.4, -2.6, 2.7, -1.5, -0.3, -0.5, -0.3, -2.2, -0.2, -5]
    bathroom_x_max = [0.4, 1.5, 1, 3.5, 3.3, 2.5, 1.5, 5.7, 1.3, 4.5, -4.2]
    bathroom_y_min = [-0.9, -1.7, -1.2, -1.5, -1.6, -1.8, -2, 0.2, -2.5, -4, 4.4]
    bathroom_y_max = [1.1, 2.2, 0.8, 0, 4, 2.3, 2, 3, 2.3, 6.5, 5.2]
    bedroom_x_min = [-1.9, -3.4, 0, -0.7, -2.4, -1.8, 3, -2.5, -1.4, -1, -5]
    bedroom_x_max = [1.7, 2, 4, 2.6, 2.2, 2.3, 5, 2.5, 3.3, 1.5, 0]
    bedroom_y_min = [-1.7, -3.9, -2.7, 0.1, -1.9, -1.6, -1.7, -3.3, -3.6, -5, -1]
    bedroom_y_max = [2, 4.9, 1, 2.7, 1.5, 2.3, 1.4, 2.5, 0, 0.6, 3.6]
    kitchen_x_min = [-2.4, -2.4, -1.6, -5, -3, -2, -6, -3.8, -2.6, -2.2, -1.8, 0.4, -2.2]
    kitchen_x_max = [1.0, 0.5, 5, 3, 5, 2, -2, -1.7, 1.4, 2, 2.5, 1.6, 1.8]
    kitchen_y_min = [-0.8, -4.2, -5.8, -4, -2, -4.8, -1.5, -1.7, -2, -2.2, -1.8, -0.93, -2.4]
    kitchen_y_max = [1.3, 0.2, -1.2, 10, 4, 2.5, 1.3, 0.9, 6, 1.8, 2, 0.57, 2.2]
    livingroom_x_min = [-1.7, -1, -2.8, -2.4, -4.7, -1.7, -2, -5, -4.5, -2, -3]
    livingroom_x_max = [1.3, 7, 2.8, 0.5, 3, 2.8, 1.5, 4, 2, 2, 0.9]
    livingroom_y_min = [-4, -1.5, -3, -3.9, -4, -2.4, -0.8, -3, -3, -2, -6]
    livingroom_y_max = [0, 0.7, 0.7, 2.4, 0, 1.8, 5, 0.7, 2.4, 2, 1]
    office_x_min = [-4, -5.7, -4, -2.4, -3, -6, 0, -6, -5, -4, -6, -1, -15, -3, -4, -1.2]
    office_x_max = [3.3, 0.3, 4, 3, -0.3, 1, 0, 1.5, 4, 3, 5, 6, -6, -0.5, 3.5, 1.4]
    office_y_min = [-1.2, 0.5, -4, -5, -2, -4, 0, 0, -1.5, -7.5, -4, -6, -6, -7.5, -5, -1.7]
    office_y_max = [5.5, 5.5, 2, 4, 3.5, 4.5, 0, 4.3, 6, 1, 3.5, 10, 2.8, -0.3, 1.3, 1]
    index = layout_obj_file.find("_")
    if layout_obj_file[2:4] == "ba":
        filenumber = int(layout_obj_file[19:index])
        x_min = bathroom_x_min[filenumber-1]
        x_max = bathroom_x_max[filenumber-1]
        y_min = bathroom_y_min[filenumber-1]
        y_max = bathroom_y_max[filenumber-1]
    if layout_obj_file[2:4] == "be":
        filenumber = int(layout_obj_file[17:index])
        x_min = bedroom_x_min[filenumber-1]
        x_max = bedroom_x_max[filenumber-1]
        y_min = bedroom_y_min[filenumber-1]
        y_max = bedroom_y_max[filenumber-1]
    if layout_obj_file[2:4] == "ki":
        filenumber = int(layout_obj_file[17:index])
        x_min = kitchen_x_min[filenumber-1]
        x_max = kitchen_x_max[filenumber-1]
        y_min = kitchen_y_min[filenumber-1]
        y_max = kitchen_y_max[filenumber-1]
    if layout_obj_file[2:4] == "li":
        filenumber = int(layout_obj_file[23:index])
        x_min = livingroom_x_min[filenumber-1]
        x_max = livingroom_x_max[filenumber-1]
        y_min = livingroom_y_min[filenumber-1]
        y_max = livingroom_y_max[filenumber-1]
    if layout_obj_file[2:4] == "of":
        filenumber = int(layout_obj_file[15:index])
        x_min = office_x_min[filenumber-1]
        x_max = office_x_max[filenumber-1]
        y_min = office_y_min[filenumber-1]
        y_max = office_y_max[filenumber-1]

    with open('config.json') as f:
        df = json.load(f)

    filename = layout_obj_file.replace("./", '')

    camera_x = random.uniform(x_min, x_max)
    camera_y = random.uniform(y_min, y_max)
    camera_z = random.uniform(1, 2.5)
    lamp_x = random.uniform(x_min, x_max)
    lamp_y = random.uniform(y_min, y_max)
    lamp_z = random.uniform(1, 2.2)

    if df[filename]["camera"].get("x") is not None:
        camera_x = df[filename]["camera"]["x"]
    if df[filename]["camera"].get("y") is not None:
        camera_y = df[filename]["camera"]["y"]

    if df[filename]["lamp"].get("x") is not None:
        lamp_x = df[filename]["lamp"]["x"]
    if df[filename]["lamp"].get("y") is not None:
        lamp_y = df[filename]["lamp"]["y"]
    if df[filename]["lamp"].get("z") is not None:
        lamp_z = df[filename]["lamp"]["z"]

    if df[filename]["at"].get("z") is not None:
        at_z = df[filename]["at"]["z"]
        camera_z = at_z + 0.3

    attention_point = [(x_max + x_min)/2, (y_max + y_min)/2, at_z]
    print("attention_point:", attention_point)
    camera_location = (camera_x, camera_y, camera_z)
    print("camera_location:", camera_location)
    camera_euler = calculate_camera_rotation(camera_location, attention_point)
    lamp_location = (lamp_x, lamp_y, lamp_z)
    print("lamp_location: ", lamp_location)
    return camera_location, camera_euler, lamp_location

def main(filename):
    layout_obj_file = filename.replace('./SceneNet', '.')
    print(layout_obj_file.replace('./', ''))
    camera_location, camera_euler, lamp_location = determine_lamp_and_camera(layout_obj_file)
    camera_info = { "location": camera_location, "euler": camera_euler }
    lamp_info = { "location" : lamp_location }
    layout_obj_file = layout_obj_file.replace("./", "./SceneNet/")
    layout_mtl_file = layout_obj_file.replace("obj", "mtl")
    format_mtl_file(layout_mtl_file)
    render_with_blender(layout_obj_file, lamp_info, camera_info)
    output_lamp_camera_information(camera_info, lamp_info, layout_obj_file.replace('./SceneNet/', ''))

if __name__ == '__main__':
    args = sys.argv
    main(args[-1])
