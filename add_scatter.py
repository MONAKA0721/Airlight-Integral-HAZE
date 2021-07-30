import array
import glob
import math
import os
import random
import sys

import cv2
import Imath
import matplotlib.pyplot as plt
import numpy as np
import OpenEXR
from PIL import Image
from tqdm import tqdm


def import_exrfile(filePath):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    img_exr = OpenEXR.InputFile(filePath)
    r_str, g_str, b_str = img_exr.channels('RGB', pt)
    z_str = img_exr.channels('Z', pt)
    red = np.array(array.array('f', r_str))
    green = np.array(array.array('f', g_str))
    blue = np.array(array.array('f', b_str))
    depth = np.array(array.array('f', z_str[0]))
    depth = depth.reshape(480, 640)
    img = np.array([[r, g, b] for r, g, b in zip(red, green, blue)])
    threshold = 10
    depth_mask = depth < threshold
    depth_mask2 = depth > threshold
    depth = depth * depth_mask + depth_mask2 * threshold
    hazefree = (img.T).flatten()
    hazefree = hazefree.reshape(3, 480, 640)
    hazefree = hazefree.astype(np.float32)
    print("depth max : ", np.max(depth))
    return hazefree, depth, img


def calc(vp_vec, dvp, sp_vec, dsp, g, samp, vs_vec, beta, resolution):
    vp_norm = vp_vec / dvp  # (3, resolution)
    # print(vp_norm)
    sp_norm = sp_vec / dsp  # (3, resolution)
    dvp_max = np.max(dvp)
    print("dv_max : ", dvp_max)

    samp_d = dvp_max / samp
    x = np.array([samp_d * i for i in range(samp)]).reshape(samp, 1)
    mask = x <= dvp_max

    # カメラから粒子までのベクトル
    vq_vec = vp_norm.T * x.reshape(samp, 1, 1)  # (samp, resolution, 3)
    dvq = np.linalg.norm(vq_vec, ord=2, axis=2).reshape(
        samp, resolution, 1)  # (samp, resolution, 1)
    dvq_nonzero = np.where(dvq != 0, dvq, 1)
    vq_norm = vq_vec / dvq_nonzero  # (samp, resolution, 3)
    vq_norm = vq_norm * (dvq != 0)

    # 光源から粒子までのベクトル
    sq_vec = vq_vec - vs_vec.reshape(1, 3)  # (samp, resolution, 3)
    dsq = np.linalg.norm(sq_vec, ord=2, axis=2).reshape(
        samp, resolution, 1)  # (samp, resolution, 1)
    dsq_nonzero = np.where(dsq != 0, dsq, 1)
    sq_norm = sq_vec / dsq_nonzero  # (samp, resolution, 3)
    sq_norm = sq_norm * (dsq != 0)

    y = np.linalg.norm(vs_vec.reshape(1, 3) - vq_vec,
                       ord=2, axis=2)  # (samp, resolution)

    cos = np.sum(-vq_norm * sq_norm, axis=2)  # (samp, resolution)
    cos_sum = np.sum(cos, axis=0, keepdims=True)

    p = (1 - np.power(g, 2)) / (np.power((1 - 2 * g * cos + g**2), 1.5))

    ans = np.exp(- beta * (x + y)) * p / (np.power(y, 2))

    extra_cos = np.sum(-vp_norm * sp_norm, axis=0,
                       keepdims=True)  # (1, resolution)
    cos_sum += extra_cos
    cos_sum = cos_sum.reshape(480, 640)

    extra_p = (1 - np.power(g, 2)) / \
        (np.power((1 - 2 * g * extra_cos + g**2), 1.5))  # (1, resolution)
    extra_ans = np.exp(- beta * (dvp + dsp)) * extra_p / \
        (np.power(dsp, 2))  # (1, resolution)

    mod = np.mod(dvp, samp_d)
    mod = mod + 0.0000001 * (mod < 0.0000001)
    extra_ans = extra_ans * mod
    ans = ans * mask * samp_d

    s = np.sum(ans, axis=0) + extra_ans
    return s, cos_sum


def output_beta_information(filePath, beta):
    s = "beta: " + str(beta)
    with open(filePath, mode='w') as f:
        f.write(s)


def depth2vp(depth):
    fovy = 61.9
    width = 640
    height = 480
    f = height / (2 * math.tan(math.radians(fovy/2)))
    c_x = width/2
    c_y = height/2
    A = np.array([[f, 0, c_x], [0, f, c_y], [0, 0, 1]])
    inv_A = np.linalg.inv(A)

    mesh = np.load('mesh.npy')
    mesh = mesh.T
    mesh[0] = mesh[0] - width/2
    mesh[1] = mesh[1] - height/2
    mesh_camera = np.dot(inv_A, mesh)
    vp_vec = mesh_camera * depth  # (3, 307200)
    return vp_vec


def calc_rot(camera_euler):
    # blenderのx軸方向に 180度 回転させることにより、デフォルトのカメラ方向からのxyz座標に直す
    rotation_to_default = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    ez = camera_euler[0]
    ey = camera_euler[1]
    ex = camera_euler[2]
    Rz = np.array([[math.cos(ez), -math.sin(ez), 0],
                  [math.sin(ez), math.cos(ez), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(ey), 0, math.sin(ey)], [
                  0, 1, 0], [-math.sin(ey), 0, math.cos(ey)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(ex), -math.sin(ex)],
                  [0, math.sin(ex), math.cos(ex)]])
    rotation_to_camera = np.dot(Ry, Rz)
    rotation_to_camera = np.dot(Rx, rotation_to_camera)
    rotation_to_camera = np.linalg.inv(rotation_to_camera)
    # rotation_to_camera = np.dot(rotation_to_camera, rotation_to_default)
    # rotation_to_camera = np.dot(rotation_to_camera, rotation_to_default)
    return rotation_to_camera


def calc_vs(camera_location, lamp_location, camera_euler):
    rotation_to_camera = calc_rot(camera_euler)
    vs_vec_on_world = (lamp_location - camera_location).reshape(3, 1)

    vs_vec = np.dot(rotation_to_camera, vs_vec_on_world)
    vs_vec[1] = -vs_vec[1]
    vs_vec[2] = -vs_vec[2]
    return vs_vec


def add_scatter(img, depth, camera_location, lamp_location, camera_euler):
    width = 640
    height = 480
    depth = depth.flatten()
    vp_vec = depth2vp(depth)  # (3, resolution)

    vs_vec = calc_vs(camera_location, lamp_location, camera_euler)  # (3, 1)

    sp_vec = vp_vec - vs_vec    # (3, resolution)
    dsp = np.linalg.norm(sp_vec, ord=2, axis=0)  # (resolution, )

    dvp = np.linalg.norm(vp_vec, ord=2, axis=0)  # (resolution, )
    # print(np.max(d1_norm), np.max(d2_norm))

    resolution = width * height

    # use below for main net dataset
    beta = random.uniform(0.1, 0.4)

    # #use below for Ls net dataset
    # beta = np.random.rand(307200).reshape(1, 307200)

    print("beta:", beta)
    g = 0.9

    # print(beta.shape)
    # 積分の結果
    s, cos_sum = calc(vp_vec, dvp, sp_vec, dsp, g, 100,
                      vs_vec, beta, resolution)  # (1, resolution)

    watt = 300
    light_intensity = watt/(4*math.pi)
    light_intensity = np.array(
        [light_intensity, light_intensity, light_intensity]).reshape(3, 1)

    img_pixels = img.T
    img_L = img_pixels
    # print(img_pixels.shape)

    # print("d1+d2 ", "min : ", np.min(d1_norm+d2_norm), " max : ", np.max(d1_norm+d2_norm))

    Ld = img_pixels * np.exp(- beta * (dsp + dvp))
    Ld = Ld.reshape(3, height, width).transpose(1, 2, 0)

    Ls = (beta * light_intensity * s) / (4 * math.pi)
    Ls = Ls.reshape(3, height, width).transpose(1, 2, 0)
    # print("Ls[0,0] : ", Ls[0,0])
    print("Ls max : ", np.max(Ls))
    print("Ld max : ", np.max(Ld))

    hazy = (Ld + Ls)

    integral = s.reshape(1, height, width)

    hazy = hazy.astype(np.float32)
    Ld = Ld.astype(np.float32)
    Ls = Ls.astype(np.float32)
    # beta = beta.reshape(1, 480, 640)
    return hazy, Ls, Ld, beta, integral, vs_vec, cos_sum


def parse_camera_and_lamp_information(filePath):
    camera_location = []
    lamp_location = []
    camera_euler = []
    with open(filePath, "r", encoding="utf-8") as f:
        for line in f:
            if line[:15] == "camera_location":
                index1 = line.find("(") + 1
                index2 = line.find(",")
                index3 = line.find(",", index2+1)
                index4 = line.find(")")
                camera_x = float(line[index1:index2])
                camera_y = float(line[index2+2:index3])
                camera_z = float(line[index3+2:index4])
                camera_location = [camera_x, camera_y, camera_z]
            if line[:13] == "lamp_location":
                index1 = line.find("(") + 1
                index2 = line.find(",")
                index3 = line.find(",", index2+1)
                index4 = line.find(")")
                lamp_x = float(line[index1:index2])
                lamp_y = float(line[index2+2:index3])
                lamp_z = float(line[index3+2:index4])
                lamp_location = [lamp_x, lamp_y, lamp_z]
            if line[:12] == "camera_euler":
                index1 = line.find(" ") + 1
                index2 = line.find(",")
                index3 = line.find(",", index2+1)
                if line[index1] == '[':
                    ex = float(line[index1+1:index2])
                    ey = float(line[index2+2:index3])
                    ez = 0
                else:
                    index_equal1 = line.find("=")
                    index_equal2 = line.find("=", index_equal1+1)
                    index_equal3 = line.find("=", index_equal2+1)
                    index4 = line.rfind(")")
                    ex = float(line[index_equal1+1:index2])
                    ey = float(line[index_equal2+1:index3])
                    ez = float(line[index_equal3+1:index4])
                camera_euler = [ez, ey, ex]
    return np.array(camera_location), np.array(lamp_location), camera_euler


def main(inputDirName, outputDirName, filename):
    print("-------------------------------------------")
    print("processing : ", filename)

    input_root = "/Volumes/WD_HDD_2TB/Dataset/"
    output_root = "/Volumes/WD_HDD_2TB/Dataset/"

    hazefree, depth, img = import_exrfile(filePath=os.path.join(
        input_root, 'EXRfiles', inputDirName, filename))
    camera_location, lamp_location, camera_euler = parse_camera_and_lamp_information(filePath=os.path.join(
        input_root, 'camera_lamp_information', inputDirName, filename.replace('exr', 'txt')))
    hazy, Ls, Ld, beta, integral, vs_vec, cos_sum = add_scatter(
        img, depth, camera_location, lamp_location, camera_euler)
    depth = depth.reshape(1, 480, 640)
    print("Hazy max : ", np.max(hazy))
    print("vs_vec : ", vs_vec)
    if(np.max(hazy) < 1 and np.max(Ld) >= 0.01):

        hazy = hazy.transpose(2, 0, 1)  # (3, 480, 640) RGB

        save_mode = "val" if random.uniform(0, 1) <= 0.1 else "train"

        array = [hazy, hazefree, depth, beta, vs_vec, Ls]
        os.makedirs(os.path.join(output_root, 'NdArray',
                    outputDirName, 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_root, 'NdArray',
                    outputDirName, 'val'), exist_ok=True)
        np.save(os.path.join(output_root, 'NdArray', outputDirName,
                save_mode, filename.replace("exr", "npy")), array)

        cv_hazefreePNG = cv2.cvtColor(
            hazefree.transpose(1, 2, 0)*255, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.join(output_root, 'HazefreePNG',
                    outputDirName, 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_root, 'HazefreePNG',
                    outputDirName, 'val'), exist_ok=True)
        cv2.imwrite(os.path.join(output_root, 'HazefreePNG', outputDirName,
                    save_mode, filename.replace("exr", "png")), cv_hazefreePNG)

        cv_hazyPNG = cv2.cvtColor(
            (hazy.transpose(1, 2, 0)*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.join(output_root, 'HazyPNG',
                    outputDirName, 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_root, 'HazyPNG',
                    outputDirName, 'val'), exist_ok=True)
        cv2.imwrite(os.path.join(output_root, 'HazyPNG', outputDirName,
                    save_mode, filename.replace("exr", "png")), cv_hazyPNG)

        cv_hazy = cv2.cvtColor(hazy.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.join(output_root, 'HazyEXR',
                    outputDirName, 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_root, 'HazyEXR',
                    outputDirName, 'val'), exist_ok=True)
        cv2.imwrite(os.path.join(output_root, 'HazyEXR',
                    outputDirName, save_mode, filename), cv_hazy)

    else:
        print("この画像は使えません")


if __name__ == '__main__':
    args = sys.argv
    main(args[-3], args[-2], args[-1])  # ex)description0.exr
