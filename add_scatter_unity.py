import array
import json
import math
import os
import random
import sys

import cv2
import Imath
import numpy as np
import OpenEXR
from nptyping import NDArray
from PIL import Image


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
    return s


def depth2vp(depth: NDArray[(480, 640), np.float64]
             ) -> NDArray[(3, 307200), np.float64]:
    fovy = 60
    width = 640
    height = 480
    f = height / (2 * math.tan(math.radians(fovy/2)))
    c_x = width/2
    c_y = height/2
    A = np.array([[f, 0, c_x], [0, f, c_y], [0, 0, 1]])
    inv_A = np.linalg.inv(A)

    mesh = np.load('mesh.npy')
    mesh = mesh.T  # (3, 307200)
    mesh[0] = mesh[0] - width/2
    mesh[1] = mesh[1] - height/2
    mesh_camera = np.dot(inv_A, mesh)
    vp_vec = mesh_camera * depth.flatten()  # (3, 307200)
    return vp_vec


def add_scatter(hazefree: NDArray[(3, 480, 640), np.float64],
                depth: NDArray[(480, 640), np.float64],
                lamp_location: NDArray[(3,), np.float64]
                ) -> tuple[NDArray[(3, 480, 640), np.float64],
                           NDArray[(480, 640), np.float64],
                           NDArray[(3, 480, 640), np.float64],
                           float,
                           NDArray[(3,), np.float64]]:
    height, width = 480, 640
    resolution = width * height

    vs_vec = lamp_location.reshape(3, 1)
    vp_vec = depth2vp(depth)
    sp_vec = vp_vec - vs_vec

    dsp = np.linalg.norm(sp_vec, ord=2, axis=0)  # (resolution, )
    dvp = np.linalg.norm(vp_vec, ord=2, axis=0)  # (resolution, )

    beta = random.uniform(0.05, 0.3)
    print("beta:", beta)
    g = 0.9

    # 積分の結果
    s = calc(vp_vec, dvp, sp_vec, dsp, g, 50,
             vs_vec, beta, resolution)  # (1, resolution)

    watt = 1000
    light_intensity = watt/(4*math.pi)
    light_intensity = np.array(
        [light_intensity, light_intensity, light_intensity]).reshape(3, 1)

    ld = hazefree * \
        np.exp(- beta * (dsp.reshape(height, width) +
                         dvp.reshape(height, width)))

    ls = (beta * light_intensity * s) / (4 * math.pi)
    ls = ls.reshape(3, height, width)

    print("Ls max : ", np.max(ls))
    print("Ld max : ", np.max(ld))

    hazy = ld + ls

    return hazy, ls, ld, beta, vs_vec


def import_png_file(file_path: str) -> NDArray[(3, 480, 640), np.float64]:
    return (np.array(Image.open(file_path))/255).transpose(2, 0, 1)[:3, :, :]


def import_depth_file(file_path: str) -> NDArray[(480, 640), np.float64]:
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    img_exr = OpenEXR.InputFile(file_path)

    # 読んだEXRファイルからRGBの値を取り出し
    r_str, g_str, b_str = img_exr.channels('RGB', pt)
    red = np.array(array.array('f', r_str))
    green = np.array(array.array('f', g_str))
    blue = np.array(array.array('f', b_str))

    # 画像サイズを取得
    dw = img_exr.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # openCVで使えるように並べ替え
    img = np.array([[r, g, b] for r, g, b in zip(red, green, blue)])
    img = img.reshape(size[1], size[0], 3)
    return img[:, :, 0]


def import_light_position_file(light_position_path: str
                               ) -> NDArray[(3,), np.float64]:
    with open(light_position_path) as f:
        df = json.load(f)
    return np.array([df['x'], df['y'], df['z']])


def main(output_dir_path, png_file_path, depth_file_path, light_position_path):
    print("-------------------------------------------")
    hazefree = import_png_file(png_file_path)
    depth = import_depth_file(depth_file_path)
    depth = np.where(depth >= 200, 200, depth)
    lamp_location = import_light_position_file(light_position_path)
    hazy, ls, ld, beta, vs_vec = add_scatter(hazefree, depth, lamp_location)
    depth = depth.reshape(1, 480, 640)
    print("Hazy max : ", np.max(hazy))

    if(np.max(hazy) < 1 and np.max(ld) >= 0.01):

        array = [hazy, hazefree, depth, beta, vs_vec, ls]

        filename = os.path.splitext(os.path.basename(png_file_path))[0]
        os.makedirs(os.path.join(output_dir_path, 'NdArray'), exist_ok=True)
        np.save(os.path.join(output_dir_path, 'NdArray',
                filename+'.npy'), array)

        cv_hazy = cv2.cvtColor((hazy*255).transpose(
            1, 2, 0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.join(output_dir_path, 'HazyPNG'), exist_ok=True)
        cv2.imwrite(os.path.join(output_dir_path,
                    'HazyPNG', filename+'.png'), cv_hazy)

    else:
        print("この画像は使えません")


if __name__ == '__main__':
    args = sys.argv
    main(args[-4], args[-3], args[-2], args[-1])
