#!/bin/sh

image=$1

mkdir -p $(pwd)/test_dir/model
mkdir -p $(pwd)/test_dir/output

chmod -R 777 $(pwd)/test_dir/model
chmod -R 777 $(pwd)/test_dir/output

rm -rf $(pwd)/test_dir/model/*
rm -rf $(pwd)/test_dir/output/*

docker run -v $(pwd)/test_dir:/opt/ml --rm ${image} train --prefix /opt/ml