#!/bin/bash

if [ $# -ne 3 ]; then
  echo "実行するには3個の引数(inputDirName, outputDirName, 処理する個数)が必要です" 1>&2
  exit 1
fi

for index in $(seq -w $3)
do
  python3 add_scatter.py $1 $2 ${index}.exr
done
