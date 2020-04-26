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

def format_mtl_file(filename, filename2):
    try:
        tmp_list = []
        firstSlash = filename.find("/")
        secondSlash = filename.find("/", firstSlash+1)
        scene = filename[firstSlash+1:secondSlash]
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                if line[:6] == "map_Kd":
                    index = line.find(" ") + 3
                    if line[-5:-1] == ".bmp":
                        end_index = line.rfind("/")
                        map_Kd = "./SceneNet" + line[index:end_index] + "/*.bmp"
                        # ./SceneNet/texture_library/cloth/*.bmp
                    else:
                        print("else")
                        map_Kd = "./SceneNet" + line[index:-1] + "/*.bmp"

                    files = glob.glob(map_Kd)
                    file = random.choice(files)
                    new_line = file.replace("./SceneNet", "map_Kd ..")
                    tmp_list.append( new_line + "\n")
                else:
                    tmp_list.append(line)

        with open(filename2, 'w', encoding="utf-8") as f:
            for line in tmp_list:
                f.write(line)

    except IOError:
        print ("Could not open the .mtl file...")

def render_with_blender(layout_file, objects, filename, lamp, camera_info):
    # bpy.ops.wm.open_mainfile(filepath="./untitled.blend")
    for item in bpy.context.scene.objects:
        bpy.context.scene.objects.unlink(item)

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

    # ライトを置く
    bpy.ops.object.lamp_add(type='POINT', location=lamp["location"])

    # レイアウトファイルのインポート
    bpy.ops.import_scene.obj(filepath=layout_file)

    for object in objects[:10]:
        a = set(bpy.data.objects)
        objpath = "/Volumes/WD_HDD_2TB/ShapeNet/" + object["object_path"] + "/models/model_normalized.obj"
        bpy.ops.import_scene.obj(filepath=objpath)
        b = set(bpy.data.objects)
        c = b - a
        M = object["transformation"]
        gl2bl = np.array([[1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0]])
        new_trans = np.dot(gl2bl, M)
        rotation_matrix = new_trans[:, :3]
        rot = Rotation.from_dcm(rotation_matrix)
        euler = rot.as_euler('xyz')
        for i in c:
            scale = object["scale"]
            # i.scale.x = scale
            # i.scale.y = scale
            # i.scale.z = scale
            i.location.x += new_trans[0][3]
            i.location.y += new_trans[1][3]
            if(new_trans[2][3] < 0):
                i.location.z = - new_trans[2][3]
            else:
                i.location.z += new_trans[2][3]
            i.location.z += 0.12
            i.rotation_euler[0] = euler[0]
            i.rotation_euler[1] = euler[1]
            i.rotation_euler[2] = euler[2]

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

    # bpy.data.scenes["Scene"].render.filepath = "../test4.jpg"
    bpy.data.scenes[0].render.resolution_x = 640
    bpy.data.scenes[0].render.resolution_y = 480
    bpy.data.scenes[0].render.resolution_percentage = 100
    bpy.data.scenes[0].render.image_settings.file_format = 'OPEN_EXR'
    bpy.data.scenes[0].render.image_settings.use_zbuffer = True
    bpy.data.scenes[0].render.image_settings.color_depth = '32'
    bpy.ops.render.render(use_viewport=True)

    now = datetime.datetime.now()
    # save_name = now.strftime('%Y%m%d_%H%M%S') + '.exr'
    save_name = "0117/" + filename[filename.find("_d") + 1:].replace("txt", "exr")
    bpy.data.images['Render Result'].save_render(filepath="./output/depth/" + save_name)

    bpy.context.scene.use_nodes = False
    bpy.ops.render.render(use_viewport=True)
    bpy.data.images['Render Result'].save_render(filepath="/Volumes/WD_HDD_2TB/Dataset/EXRfiles/" + save_name)

def parse_description(filename):
    try:
        layout_file = ""
        object_path = ""
        objects = []
        with open(filename, "r", encoding="utf-8") as f:
            count = 0
            object_str_index = -1
            scale_index = -1
            transformation_index = -5
            trans_row1 = np.array([0, 0, 0, 0])
            trans_row2 = np.array([0, 0, 0, 0])
            trans_row3 = np.array([0, 0, 0, 0])
            transformation = None
            for line in f:
                if line[:11] == "layout_file":
                    index = line.find(" ") + 1
                    layout_file = line[index:-1]
                if line[:6] == "object":
                    object_str_index = count + 1
                if line[:5] == "scale":
                    scale_index = count + 1
                if line[:14] == "transformation":
                    transformation_index = count
                if count == object_str_index:
                    object_path = line[:-1]
                if count == scale_index:
                    scale = float(line[:-1])
                    transformation = 0
                if count == transformation_index + 1:
                    index1 = line.find(" ")
                    a = float(line[:index1])
                    index2 = line.find(" ", index1 + 1)
                    b = float(line[index1+1:index2])
                    index3 = line.find(" ", index2 + 1)
                    c = float(line[index2+1:index3])
                    d = float(line[index3+1:-1])
                    trans_row1 = np.array([a, b, c, d])
                if count == transformation_index + 2:
                    index1 = line.find(" ")
                    a = float(line[:index1])
                    index2 = line.find(" ", index1 + 1)
                    b = float(line[index1+1:index2])
                    index3 = line.find(" ", index2 + 1)
                    c = float(line[index2+1:index3])
                    d = float(line[index3+1:-1])
                    trans_row2 = np.array([a, b, c, d])
                if count == transformation_index + 3:
                    index1 = line.find(" ")
                    a = float(line[:index1])
                    index2 = line.find(" ", index1 + 1)
                    b = float(line[index1+1:index2])
                    index3 = line.find(" ", index2 + 1)
                    c = float(line[index2+1:index3])
                    d = float(line[index3+1:-1])
                    trans_row3 = np.array([a, b, c, d])
                    transformation = np.stack([trans_row1, trans_row2, trans_row3])
                if line == "\n":
                    object = {"object_path":object_path, "scale":scale, "transformation":transformation}
                    objects.append(object)
                if line[0] == "#":
                    break
                count += 1
        return layout_file, objects
    except IOError:
        print ("Could not open the description file...")

def output_lamp_camera_information(camera, lamp, description_filename):
    save_name = "0117/" + description_filename[description_filename.find("_d") + 1:]
    # now = datetime.datetime.now()
    # save_name = now.strftime('%Y%m%d_%H%M%S') + '.txt'
    save_name = "/Volumes/WD_HDD_2TB/Dataset/camera_lamp_information/" + save_name
    print(save_name)
    s = "camera_location: " + str(camera["location"]) + "\n" + \
        "camera_euler(zyx): " + str(camera["euler"])  + "\n" + \
        "lamp_location: " + str(lamp["location"]) + "\n" + \
        "\n" + \
        "created_by: " + description_filename[description_filename.find("_d") + 1:]
    with open(save_name, mode='w') as f:
        f.write(s)

def calculate_camera_rotation(camera_location, attention_point, camera_location_number, is_narrow):
    camera_location = np.array([camera_location[0], camera_location[1], camera_location[2]])
    if is_narrow:
        v = np.array(attention_point) - camera_location
        r_3 = -v/np.linalg.norm(v)
        r_1 = - np.cross(r_3, np.array([0, 0, 1]))
        r_2 = np.cross(r_1, r_3)
        rotation_matrix = np.concatenate([r_1.reshape(3, -1), r_2.reshape(3, -1), r_3.reshape(3, -1)], axis=1)
        gl2bl = np.array([[1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0]])
        # rotation_matrix = np.dot(rotation_matrix,gl2bl)
        # rotation_matrix = np.linalg.inv(rotation_matrix)
        rot = Rotation.from_dcm(rotation_matrix)
        # euler = rot.as_euler('zyx')
        r = rotation_matrix
        matrix = mathutils.Matrix(((r[0][0],r[0][1], r[0][2]), (r[1][0], r[1][1], r[1][2]), (r[2][0], r[2][1], r[2][2])))
        euler = matrix.to_euler('ZYX')
        return euler
    else:
        euler_y = -1
        if camera_location_number == 0:
            euler_y = 90 + random.uniform(-30, 30)
        elif camera_location_number == 1:
            euler_y = 180 + random.uniform(-30, 30)
        elif camera_location_number == 2:
            euler_y = 270 + random.uniform(-30, 30)
        elif camera_location_number == 3:
            euler_y = 0 + random.uniform(-30, 30)
        euler_zyx = [math.radians(90), math.radians(euler_y), 0]
        return euler_zyx

def determine_lamp_and_camera(layout_obj_file):
    bathroom_x_min = [-0.6, -1.4, -2.6, 2.7, -1.5, -0.3, -0.5, -0.3, -2.2, -0.2, -5]
    bathroom_x_max = [0.4, 1.5, 0.2, 3.5, 3.3, 2.5, 1.5, 5.7, 1.3, 4.5, -4.2]
    bathroom_y_min = [-0.9, -1.7, -1.2, -1.5, -1.6, -1.8, -2, 0.2, -2.5, -4, 4.4]
    bathroom_y_max = [1.1, 2.2, 0.5, 0, 4, 2.3, 2, 3, 2.3, 6.5, 5.2]
    bedroom_x_min = [-1.9, -3.4, 0, -0.7, -2.4, -1.8, 3, -2.5, -1.4, -1, -5]
    bedroom_x_max = [1.7, 2, 4, 2.6, 2.2, 2.3, 5, 2.5, 3.3, 1.5, 0]
    bedroom_y_min = [-1.7, -3.9, -2.7, 0.1, -1.9, -1.6, -1.7, -3.3, -3.6, -5, -2.4]
    bedroom_y_max = [2, 4.9, 1, 2.7, 1.5, 2.3, 1.4, 2.5, 0, 0.6, 3.6]
    kitchen_x_min = [-2.4, -2.4, -1.6, -3.5, -3, -2, -4, -3.8, -2.6, -2.2, -1.8]
    kitchen_x_max = [1.0, 0.5, 5, 8, 2, 2, -1.5, -1.7, 1.4, 2, 2.5]
    kitchen_y_min = [-0.8, -4.2, -5.8, -6, -13, -4.8, -1.5, -1.7, -4.3, -2.2, -1.8]
    kitchen_y_max = [1.3, 0.2, -1.2, 4, -6, 2.5, 1.3, 0.9, 3.4, 1.8, 2]
    livingroom_x_min = [-1.7, -3.5, -2.8, -2.4, -4.7, -1.7, -2, -5, -4.5, -2, -3]
    livingroom_x_max = [1.3, 7, 2.8, 0.5, 3, 2.8, 1.5, 4, 2, 2, 0.9]
    livingroom_y_min = [-4, -1.5, -3, -3.9, -4, -2.4, -0.8, -3, -3, -2, -6]
    livingroom_y_max = [0, 0.7, 0.7, 2.4, 0, 1.8, 5, 0.7, 2.4, 2, 1]
    office_x_min = [-8.8, -5.7, -4, -2.4, -3, -6, 0, -6, -5, -4, -6, -1, -15, -3, -4]
    office_x_max = [3.3, 0.3, 4, 3, -0.3, 1, 0, 1.5, 4, 3, 5, 6, -6, -0.5, 3.5]
    office_y_min = [-1.2, 0.5, -1, -5, -2, -4, 0, 0, -1.5, -7.5, -4, -6, -6, -7.5, -5]
    office_y_max = [5.5, 5.5, 4.3, 4, 3.5, 4.5, 0, 4.3, 6, 1, 3.5, 10, 2.8, -0.3, 1.3]
    x_max = 2
    x_min = -2
    y_max = 2
    y_min = -2
    index = layout_obj_file.find("_")
    filenumber = 0
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
    print(filenumber)
    x_range = x_max - x_min
    y_range = y_max - y_min
    is_narrow = False
    is_x_narrow = False
    is_y_narrow = False
    if x_range < 3:
        is_x_narrow = True
    if y_range < 3:
        is_y_narrow = True
    first = random.choice(["x", "y"])
    camera_location_number = -1
    portion = 2/5
    if first == "x":
        camera_x = random.uniform(x_min+ x_range * portion, x_min + x_range*(1-portion))
        if camera_x == x_min+x_range*portion or camera_x == x_min + x_range*(1-portion):
            camera_y = random.uniform(y_min+y_range*portion, y_min + y_range*(1-portion))
        else:
            camera_y = random.choice([y_min+y_range*portion, y_min + y_range*(1-portion)])
    else:
        camera_y = random.uniform(y_min+y_range* portion, y_min + y_range* (1-portion))
        if camera_y == y_min+y_range* portion or camera_y == y_min + y_range* (1-portion):
            camera_x = random.uniform(x_min+x_range* portion, x_min + x_range* (1-portion))
        else:
            camera_x = random.choice([x_min+x_range* portion, x_min+ x_range * (1-portion)])
    if camera_x == x_min + x_range* portion:
        camera_location_number = 0
    elif camera_y == y_min + y_range* portion:
        camera_location_number = 1
    elif camera_x == x_min + x_range * (1-portion):
        camera_location_number = 2
    elif camera_y == y_min + y_range* (1-portion):
        camera_location_number = 3
    camera_z = random.uniform(1, 1.5)
    lamp_portion = (portion+0.5)/2
    if camera_x <= x_min + x_range/2 and camera_y <= y_min + y_range/2:
        lamp_x = random.uniform(x_min + x_range * lamp_portion, x_min + x_range*0.5)
        lamp_y = random.uniform(y_min + y_range * lamp_portion, y_min + y_range*0.5)
    elif camera_x <= x_min + x_range/2 and camera_y >= y_min + y_range/2:
        lamp_x = random.uniform(x_min + x_range * lamp_portion, x_min + x_range*0.5)
        lamp_y = random.uniform(y_min + y_range*0.5, y_min + y_range * (1-lamp_portion))
    elif camera_x >= x_min + x_range/2 and camera_y >= y_min + y_range/2:
        lamp_x = random.uniform(x_min + x_range*0.5, x_min + x_range * (1-lamp_portion))
        lamp_y = random.uniform(y_min + y_range*0.5, y_min + y_range * (1-lamp_portion))
    elif camera_x >= x_min + x_range/2 and camera_y <= y_min + y_range/2:
        lamp_x = random.uniform(x_min + x_range*0.5, x_min + x_range * (1-lamp_portion))
        lamp_y = random.uniform(y_min + y_range * lamp_portion, y_min + y_range*0.5)
    if is_x_narrow and camera_location_number == 0:
        is_narrow = True
        lamp_x = random.uniform(x_min, x_min + x_range*portion)
    elif is_x_narrow and camera_location_number == 2:
        is_narrow = True
        lamp_x = random.uniform(x_max, x_min + x_range*(1-portion))
    elif is_y_narrow and camera_location_number == 1:
        is_narrow = True
        lamp_y = random.uniform(y_min, y_min + y_range*portion)
    elif is_y_narrow and camera_location_number == 3:
        is_narrow = True
        lamp_y = random.uniform(y_max, y_min + y_range*(1-portion))
    lamp_z = min(2, max(1, camera_z + random.uniform(-0.5, 0.5)))
    attention_point = [(x_max + x_min)/2, (y_max + y_min)/2, 1.25]
    camera_location = (camera_x, camera_y, camera_z)
    print(camera_location_number)
    camera_euler = calculate_camera_rotation(camera_location, attention_point, camera_location_number, is_narrow)
    lamp_location = (lamp_x, lamp_y, lamp_z)
    return camera_location, camera_euler, lamp_location

def main(filename):
    layout_obj_file, objects = parse_description(filename)
    camera_location, camera_euler, lamp_location = determine_lamp_and_camera(layout_obj_file)
    camera_info = { "location": camera_location, "euler": camera_euler }
    lamp_info = { "location" : lamp_location }
    layout_obj_file = layout_obj_file.replace("./", "./SceneNet/")
    layout_mtl_file = layout_obj_file.replace("obj", "mtl")
    format_mtl_file(layout_mtl_file, layout_mtl_file)
    render_with_blender(layout_obj_file, objects, filename, lamp_info, camera_info)
    output_lamp_camera_information(camera_info, lamp_info, filename)

if __name__ == '__main__':
    args = sys.argv
    main(str(args[-1]))
