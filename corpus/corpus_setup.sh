#!/bin/bash

thread=$1

cd corpus

bash corpus_pubmed_util.sh 1 100 1 &
bash corpus_pubmed_util.sh 101 200 2 &
bash corpus_pubmed_util.sh 201 300 3 &
bash corpus_pubmed_util.sh 301 400 4 &
bash corpus_pubmed_util.sh 401 500 5 &
bash corpus_pubmed_util.sh 501 600 6 &
bash corpus_pubmed_util.sh 601 700 7 &
bash corpus_pubmed_util.sh 701 800 8 &
bash corpus_pubmed_util.sh 801 900 9 &
bash corpus_pubmed_util.sh 901 1000 10 &
bash corpus_pubmed_util.sh 1001 1015 11 &

wait

cat *.tsv > pubmed.tsv
rm *.log
rm pubmed2*

wget http://cs.iupui.edu/~phillity/ade.tsv
wget http://cs.iupui.edu/~phillity/cdr.tsv
wget http://cs.iupui.edu/~phillity/yahoo_qa.tsv

cd ..

bash corpus/dataset_parse.sh cdr $thread
bash corpus/dataset_parse.sh yahoo_qa $thread
bash corpus/dataset_parse.sh pubmed $thread
