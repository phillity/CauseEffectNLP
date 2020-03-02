#!/bin/bash

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
fi=pubmed.tsv
num_files=64
total_lines=$(wc -l <${fi})
((lines_per_file = (total_lines + num_files - 1) / num_files))
split -d --lines=${lines_per_file} ${fi} --additional-suffix=.tsv pubmed_
rm pubmed.tsv


wget http://cs.iupui.edu/~phillity/yahoo_qa.tsv
fi=yahoo_qa.tsv
num_files=64
total_lines=$(wc -l <${fi})
((lines_per_file = (total_lines + num_files - 1) / num_files))
split -d --lines=${lines_per_file} ${fi} --additional-suffix=.tsv yahoo_qa_
rm yahoo_qa.tsv

wget http://cs.iupui.edu/~phillity/cdr.tsv
fi=cdr.tsv
num_files=64
total_lines=$(wc -l <${fi})
((lines_per_file = (total_lines + num_files - 1) / num_files))
split -d --lines=${lines_per_file} ${fi} --additional-suffix=.tsv cdr_
rm cdr.tsv
