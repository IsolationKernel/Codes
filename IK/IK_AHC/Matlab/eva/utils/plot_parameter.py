# Copyright 2022 Xin Han. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# setting
dataname = "thyroid"
y_label = 'F1 Score'  # 'F1 Score', 'purity'
score_type = 'f1_score'  # 'f1_score', 'Dendrogram Purity'
period = 10

file_path = './exp_out/khc/repeat_para/'+score_type+'.csv'
save_path = './exp_out/khc/repeat_para/'+score_type


os.makedirs(save_path, exist_ok=True)

data = pd.read_csv(file_path, header=None, names=[
                   'data', 'alg', 'kernel', 'psi', 'value'])

subdata = data[data["data"] == dataname]
dfm = subdata.groupby(['alg', 'psi']).agg([np.mean, np.std])

plt.style.use(['science', 'ieee', 'retro'])
stack_dfm = dfm["value"].unstack(level=0)

x = stack_dfm[::period].index.tolist()
pha_y = stack_dfm[::period]['mean'][' PHA'].tolist()
pha_yerr = stack_dfm[::period]['std'][' PHA'].tolist()

average_y = stack_dfm[::period]['mean'][' average'].tolist()
average_yerr = stack_dfm[::period]['std'][' average'].tolist()

single_y = stack_dfm[::period]['mean'][' single'].tolist()
single_yerr = stack_dfm[::period]['std'][' single'].tolist()


data_1 = {
    'label': 'PHA',
    'x': x,
    'y': pha_y,
    'yerr': pha_yerr,
    'fmt': ':',
    'color': '#e770a2',
}

data_2 = {
    'label': 'Average',
    'x': x,
    'y': average_y,
    'yerr': average_yerr,
    'fmt': '-.',
    'color': '#f79a1e',
}

data_3 = {
    'label': 'Single',
    'x': x,
    'y': single_y,
    'yerr': single_yerr,
    'fmt': '--',
    'color': '#4165c0',
}


for data in [data_3, data_1, data_2]:
    plt.ylim([0.0, 1.0])
    plt.errorbar(**data, alpha=.75, capsize=3, capthick=1)
    data = {
        'x': data['x'],
        'y1': [y - e for y, e in zip(data['y'], data['yerr'])],
        'y2': [y + e for y, e in zip(data['y'], data['yerr'])],
        'facecolor': data['color']}
    plt.fill_between(**data, alpha=.15)

plt.xlabel('$\psi$')
plt.ylabel(ylabel=y_label)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(save_path, dataname+'_bar.pdf'))
