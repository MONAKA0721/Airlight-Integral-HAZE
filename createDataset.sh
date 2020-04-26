#!/bin/bash

for index in $(seq 10 15)
do
  S=$(printf "%04d" "${index}")
  /Applications/Blender.app/Contents/MacOS/blender -b -P blender.py "/Volumes/WD_HDD_2TB/Description/scene_and_trajectory_description${S}.txt"
done
