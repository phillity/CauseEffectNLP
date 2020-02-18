#!/bin/bash

for i in {48..63}
do
  ( python3 src/parse.py -i "pubmed_"$i".tsv" ) &
done
wait
