#!/usr/bin/env bash

set -xu

exp_dir=$1

find $exp_dir -name score.tsv -exec tail -n +2 {} \; > $exp_dir/all_scores.txt

python3 bin/util/format_result_table.py $exp_dir/all_scores.txt > $exp_dir/dendrogram_purity.tex

cat $exp_dir/dendrogram_purity.tex

echo "Dendrogram Purity Result table saved here: $exp_dir/dendrogram_purity.tex"