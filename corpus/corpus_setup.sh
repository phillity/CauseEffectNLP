#!/bin/bash


fi=pubmed.tsv
num_files=64

total_lines=$(wc -l <${fi})
((lines_per_file = (total_lines + num_files - 1) / num_files))

split -d --lines=${lines_per_file} ${fi} --additional-suffix=.tsv pubmed_

rm pubmed.tsv
