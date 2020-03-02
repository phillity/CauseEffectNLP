#!/bin/bash

dataset=$1

for i in {00..63}
do
  ( python3 src/parse.py -i $dataset"_"$i".tsv" ) &
done
wait
