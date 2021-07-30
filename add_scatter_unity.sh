#!/bin/bash

files="/home/yuya/IiyamaLab/0718/HF/*"

for filepath in $files; do
  depthpath=${filepath/HF/DEPTH}
  depthpath=${depthpath/png/exr}
  lightPositionPath=${filepath/HF/LIGHT_POSITION}
  lightPositionPath=${lightPositionPath/png/json}
  python add_scatter_unity.py '../0718' $filepath $depthpath $lightPositionPath
  echo $filepath
  echo $depthpath
  echo $lightPositionPath
done
