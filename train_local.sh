#!/bin/sh

image=$1

mkdir -p $(pwd)/test_dir/model
mkdir -p $(pwd)/test_dir/output

chmod -R 777 $(pwd)/test_dir/model
chmod -R 777 $(pwd)/test_dir/output

rm -rf $(pwd)/test_dir/model/*
rm -rf $(pwd)/test_dir/output/*

# Capture the output of the docker run command
output=$(docker run -v $(pwd)/test_dir:/opt/ml --rm ${image} train --prefix /opt/ml --episodes 10 --collect_episodes_per_iteration 20 --num_eval_episodes 5 --number_of_parallel_envs 10 --validation_episode 2 --use_virtual_display)

# Print the output for verification
echo "$output"

# Check if the output contains the expected result
if echo "$output" | grep -q "SUCCESS"; then
  echo "Test passed."
  exit 0
else
  echo "Test failed."
  exit 1
fi