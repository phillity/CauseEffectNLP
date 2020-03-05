#!/bin/bash

dataset=$1
thread=$2

if [ "$dataset" = "yahoo_qa" ] || [ "$dataset" = "cdr" ] || [ "$dataset" = "pubmed" ]
then
  total_lines=$(wc -l <"corpus/"${dataset}".tsv")
  ((lines_per_file = (total_lines + $thread - 1) / $thread))
  split -d --lines=${lines_per_file} "corpus/"${dataset}".tsv" --additional-suffix=.tsv $dataset"_"

  for i in {00..$((10#$thread))}
  do
    ( python3 data/dataset_parse.py -i $dataset"_"$i".tsv" ) &
  done
  wait

  cat corpus/$dataset"_**_pos.tsv" >> corpus/$dataset"_pos.tsv"
  rm corpus/$dataset"_**_pos.tsv"
  cat corpus/$dataset"_**_neg.tsv" >> corpus/$dataset"_neg.tsv"
  rm corpus/$dataset"_**_neg.tsv"
  cat corpus/$dataset"_**_pos_neg.tsv" >> corpus/$dataset"_pos_neg.tsv"
  rm corpus/$dataset"_**_pos_neg.tsv"
  rm corpus/$dataset"_**.tsv"
fi

python3 corpus/dataset_split.py -d $dataset
