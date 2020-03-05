#!/bin/bash

cd model

wget https://tfhub.dev/google/universal-sentence-encoder-large/5?tf-hub-format=compressed
var=$(pwd)
mkdir 5
tar xvzf 5?tf-hub-format=compressed -C $var'/5/'
rm 5?tf-hub-format=compressed
