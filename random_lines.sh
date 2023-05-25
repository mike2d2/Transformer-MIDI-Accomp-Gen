#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 input_filename output_filename n" >&2
  exit 1
fi

if ! [ -f "$1" ]; then
  echo "$1 is not a file" >&2
  exit 1
fi

n=$3
total_lines=$(wc -l < "$1")
if [ "$n" -gt "$total_lines" ]; then
  echo "n is greater than the number of lines in the file" >&2
  exit 1
fi

jot -r "$n" 1 "$total_lines" | sort -n | sed -n "$(echo '1,'$n'p')" "$1" > "$2"

