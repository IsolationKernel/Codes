import pandas as pd
df1 = pd.read_csv("./datasets/MoveBank/White-bearded wildebeest in Kenya-reference-data.csv")
df2 = pd.read_csv("./datasets/MoveBank/White-bearded wildebeest in Kenya-gps.csv")
df2_indexes  = list(df2['tag-local-identifier'].value_counts().index)
df2[df2_indexes[0] ==df2['tag-local-identifier']]
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
for i in range(36):
    id = list(results_.keys())[i]
    print(id)
    day = results_[id].copy()
    day['timestamp'] = day['timestamp'].map(lambda x:x[:4])
    date_set = sorted(list(set(day['timestamp'])))
    result_ = dict()
    print(date_set)
    for date in date_set:
        print(date)
        result_[date] = day[day['timestamp']==date]
        result[id] = result_
        res = result[id][date]
        result_tmp = []
        for idx,row in res.iterrows():
            result_tmp.append([row['location-long'],row['location-lat']])
        if 0 == len(result_tmp):
            print(i,date)
        traj.append(result_tmp)
