#!/bin/bash

source activate nlp

cd ..

for i in {00..63}
do
  ( python src/parse.py -i "corpus/pubmed_"$i ) &
done
wait
