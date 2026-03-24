#!/bin/sh

set +e
for exp in "$@"; do

  rm -rf "$exp/imgs"
  rm -f "$exp"/mean_*.csv
  rm -rf "$exp"/parsed_*/
  echo "done: $exp"
done
