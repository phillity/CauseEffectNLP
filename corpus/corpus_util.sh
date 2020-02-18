#!/bin/bash

start=$1
stop=$2
pid=$3

for i in $(seq -f "%04g" $start $stop)
do
  wget "ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed20n"$i".xml.gz" >> $pid".log" 2>&1
  gunzip "pubmed20n"$i".xml.gz" >> $pid".log" 2>&1
  cd ..
  python3 corpus/clean.py -i "pubmed20n"$i".xml" >> "corpus/"$pid".log" 2>&1
  cd corpus
done
