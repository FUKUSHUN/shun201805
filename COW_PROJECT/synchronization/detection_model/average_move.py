import os, sys
import csv
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

def shape_data(df):
    """ データを等間隔に整える """
    score_day_dict = {}
    date = df['start'][0]
    date = datetime.datetime(date.year, date.month, date.day, 12, 0, 0)
    end_t = date + datetime.timedelta(hours=21)
    time_list = []
    score_list = []
    for _, row in df.iterrows():
        start = row[0]
        end = row[1]
        if (end_t <= start): # 1 日の開始
            dic = dict(zip(time_list, score_list))
            score_day_dict[date] = dic
            time_list = []
            score_list = []
            date += datetime.timedelta(days=1)
            end_t = date + datetime.timedelta(hours=21)
        # delta 分当たりのスコアに直し，時刻を均等に割り振る
        average = row[3] / int(row[4]) * 60 * delta # delta 分ごとに平滑化
        dt = start
        while (dt < end):
            time_list.append(dt)
            score_list.append(average)
            dt += datetime.timedelta(minutes=delta)
    return score_day_dict

def calculate_moving_average(score_day_dict, slide_interval, average_window):
    """ スコアの移動平均をとる
        slide_interval:      スライド間隔 [minute]
        average_window:    平均をとる間隔 [minute] """
    idx_times = []
    smoothed_scores = []
    for key in score_day_dict.keys():
        score_dic = score_day_dict[key]
        df = pd.DataFrame(list(score_dic.items()), columns=['time', 'score'])
        time = df['time'][0]
        day_end = df['time'][len(df) - 1]
        while (time < day_end):
            end = time + datetime.timedelta(minutes=average_window)
            idx_times.append(time)
            extracted_df = df[(time <= df['time']) & (df['time'] < end)]
            smoothed_score = extracted_df['score'].values.sum() / len (extracted_df) if len (extracted_df) != 0 else 0
            smoothed_scores.append(smoothed_score)
            time += datetime.timedelta(minutes=slide_interval)
    return idx_times, smoothed_scores

if __name__ == "__main__":
    filenames = ['20122', '20129', '20158', '20192', '20267']
    dirname = "./test/"
    delta = 2 # 平滑化の間隔
    for filename in filenames:
        filepath = dirname + filename + '/' + filename + '_5.xlsx'
        df = pd.read_excel(filepath, header=None, names=['start', 'end', 'dict', 'score', 'total_second'], usecols=[0, 1, 2, 3, 4], sheet_name=filename)
        score_day_dict = shape_data(df)
        time_list, score_list = calculate_moving_average(score_day_dict, 5, 5)
    
        df = pd.DataFrame(zip(time_list, score_list), columns=['time', 'score'])
        df.to_csv(dirname + str(filename) + '/tmp_6.csv')
        # fig = plt.figure(figsize=(16, 8))
        # ax = fig.add_subplot(1,1,1)
        # ax.bar(time_list, score_list, width=0.1)
        # plt.show()
        # plt.savefig(dirname + filename + '/' + filename + '.jpg')
        # plt.close()
