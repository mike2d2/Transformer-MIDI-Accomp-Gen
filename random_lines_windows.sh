#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 input_file output_file N"
  exit 1
fi

input_file=$1
output_file=$2
N=$3

if [ ! -e "$input_file" ]; then
  echo "Input file does not exist: $input_file"
  exit 1
fi

if [ -e "$output_file" ]; then
  echo "Output file already exists: $output_file"
  exit 1
fi

total_lines=$(wc -l < "$input_file")
if [ "$total_lines" -lt "$N" ]; then
  echo "Error: N is greater than the total number of lines in the input file"
  exit 1
fi

sort -R "$input_file" | head -n "$N" > "$output_file"
echo "Selected $N random lines from $input_file and wrote them to $output_file"
