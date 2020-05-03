#!/bin/bash

objs=(`find . -type f -name "*.obj"`)
num_objs=${#objs[*]}

for index in $(seq 1 9999)
do
  filename=(`echo ${objs[$((RANDOM%num_objs))]}`)
  /Applications/Blender.app/Contents/MacOS/Blender -b -P blender.py ${filename}
done
