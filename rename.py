import os
import shutil
from glob import glob

from natsort import natsorted

root = "/home/yuya/.config/unity3d/Unity Technologies/Unity Simulation Smart Camera Outdoor/"
files = os.listdir(root)
dirs = [f for f in files if os.path.isdir(os.path.join(root, f))]

# for i, dir in enumerate(sorted(dirs)):
#     d = os.path.join(root, dir, "*/*.exr")
#     for index, filename in enumerate(natsorted(glob(d))):
#         # if 'lightPosition_0.json' in filename or 'lightPosition_1.json' in filename:
#         # if '_RegularCam_depth_0.exr' in filename or '_RegularCam_depth_1.exr' in filename:
#         #     print(filename)
#         #     os.remove(filename)
#         print(filename)
#         print(f"_RegularCam_depth_{index+2}")
#         print(dir + str(index).zfill(3))
#         os.rename(filename, filename.replace(
#             f"_RegularCam_depth_{index+2}", dir + str(index).zfill(3)))

# for i, dir in enumerate(sorted(dirs)):
#     d = os.path.join(root, dir, "ScreenCapture/*.json")
#     for index, filename in enumerate(natsorted(glob(d))):
#         # if 'lightPosition_0.json' in filename or 'lightPosition_1.json' in filename:
#         # if '_RegularCam_depth_0.exr' in filename or '_RegularCam_depth_1.exr' in filename:
#         #     print(filename)
#         #     os.remove(filename)
#         print(filename)
#         print(f"lightPosition_{index+2}")
#         print(dir + str(index).zfill(3))
#         os.rename(filename, filename.replace(
#             f"lightPosition_{index+2}", dir + str(index).zfill(3)))


# for i, dir in enumerate(sorted(dirs)):
#     d = os.path.join(root, dir, "*/*.png")
#     for index, filename in enumerate(natsorted(glob(d))):
#         # if 'lightPosition_0.json' in filename or 'lightPosition_1.json' in filename:
#         # if '_RegularCam_depth_0.exr' in filename or '_RegularCam_depth_1.exr' in filename:
#         #     print(filename)
#         #     os.remove(filename)
#         print(filename)
#         print(f"rgb_{index+2}")
#         print(dir + str(index).zfill(3))
#         os.rename(filename, filename.replace(
#             f"rgb_{index+2}", dir + str(index).zfill(3)))


# root = "/home/yuya/.config/unity3d/Unity Technologies/Unity Simulation Smart Camera Outdoor/*/*/*.png"
# for filename in glob(root):
#     print(filename)
#     shutil.move(filename, '/home/yuya/IiyamaLab/0718/HF/')

# root = "/home/yuya/.config/unity3d/Unity Technologies/Unity Simulation Smart Camera Outdoor/*/*/*.exr"
# for filename in glob(root):
#     print(filename)
#     shutil.move(filename, '/home/yuya/IiyamaLab/0718/DEPTH/')


root = "/home/yuya/.config/unity3d/Unity Technologies/Unity Simulation Smart Camera Outdoor/*/ScreenCapture/*.json"
for filename in glob(root):
    print(filename)
    shutil.move(filename, '/home/yuya/IiyamaLab/0718/LIGHT_POSITION/')
