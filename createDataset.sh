#!/bin/bash

if [ $# -ne 1 ]; then
  echo "実行するには1個の引数(何個画像を生成するか)が必要です" 1>&2
  exit 1
fi

objs=(`find . -type f -name "*.obj"`)
num_objs=${#objs[*]}

for index in $(seq 1 $1)
do
  filename=(`echo ${objs[$((RANDOM%num_objs))]}`)
  /Applications/Blender.app/Contents/MacOS/Blender -b -P blender.py ${filename}
done
