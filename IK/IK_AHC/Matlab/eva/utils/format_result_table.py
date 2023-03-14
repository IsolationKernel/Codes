# Copyright 2022 Xin Han. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np
import sys


def load_result_file(fn):
    """
    Result file format:
        algorithm <tab> dataset <tab> dendrogram purity
    Args:
        fn: filename
    Returns: dictionary: alg -> dataset -> (mean(dp),std(dp)
    """
    alg2dataset2score = {}
    alg2mean = {}
    with open(fn, 'r') as fin:
        for line in fin:
            try:
                splt = line.strip().split("\t")
                dataset, alg, dp = splt[:3]
                if alg not in alg2dataset2score:
                    alg2dataset2score[alg] = {}
                    alg2mean[alg] = {}
                if dataset not in alg2dataset2score[alg]:
                    alg2dataset2score[alg][dataset] = []
                alg2dataset2score[alg][dataset].append(float(dp))
            except:
                pass
    for alg in alg2dataset2score:
        mean_score = np.mean([alg2dataset2score[alg][data]
                             for data in alg2dataset2score[alg]])
        alg2mean[alg]["_mean_score"] = mean_score

    for alg in alg2dataset2score:
        for dataset in alg2dataset2score[alg]:
            mean = np.mean(alg2dataset2score[alg][dataset])
            std = np.std(alg2dataset2score[alg][dataset])
            alg2dataset2score[alg][dataset] = (mean, std)
    return (alg2dataset2score, alg2mean)


def escape_latex(s):
    s = s.replace("_", "\\_")
    return s


def latex_table(alg_score):
    table_string = """\\begin{table}\n\\centering\n\\caption{some caption}\n\\begin{tabular}"""
    # num_ds = max([len(alg2dataset2score[x]) for x in alg2dataset2score])
    alg2dataset2score = alg_score[0]
    alg2mean = alg_score[1]
    num_al = len(alg2dataset2score)
    formatting = "{c" + "c" * num_al + "}"
    table_string += format(formatting)
    table_string += "\n\\toprule\n"
    alg_names = alg2dataset2score.keys()

    ds_names = list(
        set([name for x in alg2dataset2score for name in alg2dataset2score[x]]))
    table_string += "\\bf Dataset & \\bf " + \
        " & \\bf ".join(escape_latex(x) for x in alg_names)
    table_string += " \\\\ \midrule\n"
    ds_names = sorted(ds_names)
    socre_mean = [alg2mean[alg]["_mean_score"] for alg in alg_names]
    for ds in ds_names:
        scores = ["%.2f $\\pm$ %.2f" % (alg2dataset2score[alg][ds][0], alg2dataset2score[alg][ds][1])
                  if ds in alg2dataset2score[alg] else "-" for alg in alg_names]
        table_string += "%s & %s \\\\\n" % (ds, " & ".join(scores))
    table_string += "\midrule\n"
    table_string += "bf Mean & " + \
        " & ".join("{:.2f}".format(x) for x in socre_mean)
    table_string += " \\\\ \\bottomrule\n\\end{tabular}\n\\end{table}"
    return table_string


if __name__ == "__main__":
    print(latex_table(load_result_file(sys.argv[1])))