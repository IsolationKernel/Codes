# import numpy as np
# import pandas as pd
# import datetime
# import time
# from codes.Algorithm2.idk_sub import *
# import  os
# from sklearn.metrics import roc_auc_score
# import scipy.io as scio

# # load data
# df1 = pd.read_csv("./datasets/MoveBank/Grey-headed flying fox, Adelaide, 2015-2018-reference-data.csv")
# df2 = pd.read_csv("./datasets/MoveBank/Grey-headed flying fox, Adelaide, 2015-2018.csv")
# df2_indexes = list(df2['tag-local-identifier'].value_counts().index)
# df2[df2_indexes[0] == df2['tag-local-identifier']]


# def split_data(df, distinct_col='tag-local-identifier', cols_selection=['timestamp', 'location-long', 'location-lat']):
#     df_indexes = list(df[distinct_col].value_counts().index)
#     results = dict()
#     for index in df_indexes:
#         df_tmp = df[df[distinct_col] == index]
#         results[index] = df_tmp[cols_selection]
#     return results


# results_ = split_data(df2)
# result = dict()
# traj = []
# trajt = []
# for i in range(4):
#     id = list(results_.keys())[i]
#     # print(id)
#     day = results_[id].copy()
#     day['timestamp_'] = day['timestamp'].copy()
#     day['timestamp'] = day['timestamp'].map(lambda x: x[:10])

#     date_set = sorted(list(set(day['timestamp'])))
#     result_ = dict()
#     for date in date_set:
#         # print(date)
#         result_[date] = day[day['timestamp'] == date]
#         result[id] = result_
#         res = result[id][date]
#         result_tmp = []
#         cmd_time = []
#         for idx, row in res.iterrows():
#             result_tmp.append([row['location-long'], row['location-lat']])
#             cmd_time.append([row['timestamp_']])
#         if 0 == len(result_tmp):
#             print(i, date)
#         if 500 < len(result_tmp):
#             traj.append(result_tmp)
#             trajt.append(cmd_time)


# def get_timestamp(str_time):
#     tmp_ = time.strptime(str_time[:-4], "%Y-%m-%d %H:%M:%S")
#     tmp_ = time.mktime(tmp_)
#     return tmp_


# def convert_traj_time_from_start(time_str_):
#     time_tmp = [get_timestamp(x) for x in time_str_]
#     time_tmp_start = time_tmp[0]
#     time_tmp = [x - time_tmp_start for x in time_tmp]
#     return time_tmp


# result_tmp = []
# for idx, traj_ in enumerate(trajt):  # 遍历每一条轨迹
#     time_list = [x[0] for x in traj_]  # 获取时间列表
#     # print(time_list)
#     time_from_start_list = convert_traj_time_from_start(time_list)
#     result_tmp.append(time_from_start_list)
# traj_time = result_tmp

# anomoly_flying_fox = [1, 21, 24, 26, 34, 38, 41, 45, 46, 47, 48]
# labels = np.ones(62)
# for i in range(len(anomoly_flying_fox)):
#     labels[anomoly_flying_fox[i]] = 0


# # algorithm2
# psi1 = 2
# t1=100
# psi2 = 2
# t2 = 100
# #print(len(traj))
# sort_y,t,t_3 = SD_diff(traj, psi1, t1,psi2, t2, 0)
# sub = subtraj(sort_y,t_3)
# print(sub)
