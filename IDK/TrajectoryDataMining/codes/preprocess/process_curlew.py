import pandas as pd
import numpy as np
import time
df1 = pd.read_csv("./datasets/MoveBank/FTZ Migrating curlews (data from Schwemmer et al. 2021)-reference-data.csv")
df2 = pd.read_csv("./datasets/MoveBank/FTZ Migrating curlews (data from Schwemmer et al. 2021).csv")
df2_indexes  = list(df2['tag-local-identifier'].value_counts().index)
df2[df2_indexes[0] == df2['tag-local-identifier']]
def split_data(df,distinct_col='tag-local-identifier',cols_selection=['timestamp','location-long','location-lat']):
    df_indexes  = list(df[distinct_col].value_counts().index)
    results = dict()
    for index in df_indexes:
        df_tmp  = df[df[distinct_col]==index]
        results[index] = df_tmp[cols_selection]
    return results
results_ = split_data(df2)
result = dict()
traj=[]
trajt=[]
for i in range(23):
    id = list(results_.keys())[i]
    #print(id)
    day = results_[id].copy()
    day['timestamp_'] = day['timestamp'].copy()
    day['timestamp'] = day['timestamp'].map(lambda x:x[:4])
    
    date_set = sorted(list(set(day['timestamp'])))
    result_ = dict()
    for date in date_set:
        #print(date)
        result_[date] = day[day['timestamp']==date]
        result[id] = result_
        res = result[id][date]
        result_tmp = []
        cmd_time = []
        for idx,row in res.iterrows():
            result_tmp.append([row['location-long'],row['location-lat']])
            cmd_time.append([row['timestamp_']])
        if 0 == len(result_tmp):
            print(i,date)
        traj.append(result_tmp)
        trajt.append(cmd_time)
def get_timestamp(str_time):
    tmp_ = time.strptime(str_time[:-4],"%Y-%m-%d %H:%M:%S")
    tmp_ = time.mktime(tmp_)
    return tmp_
def convert_traj_time_from_start(time_str_):
    time_tmp = [get_timestamp(x) for x in time_str_]
    time_tmp_start = time_tmp[0]
    time_tmp = [x - time_tmp_start for x in time_tmp]
    return time_tmp
result_tmp = []
for idx,traj_ in enumerate(trajt):#遍历每一条轨迹
    time_list =[x[0] for x in traj_] #获取时间列表
    #print(time_list)
    time_from_start_list = convert_traj_time_from_start(time_list)
    result_tmp.append(time_from_start_list)
result_tmp = []
for idx,traj_ in enumerate(trajt):#遍历每一条轨迹
    time_list =[x[0] for x in traj_] #获取时间列表
    #print(time_list)
    time_from_start_list = convert_traj_time_from_start(time_list)
    result_tmp.append(time_from_start_list)
traj_time = []
for i in range(42):
    l = len(result_tmp[i])
    cmd = []
    for j in range(l):
        cmd.append([result_tmp[i][j]])
    traj_time.append(cmd)
anomoly_curlews = [12, 25, 28, 31, 33, 34, 38, 39, 40]
labels = np.ones(42)
for i in range(len(anomoly_curlews)):
    labels[anomoly_curlews[i]] = 0
