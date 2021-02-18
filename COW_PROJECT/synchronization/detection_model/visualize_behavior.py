import os, sys
import csv
import math
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

# 自作クラス
os.chdir('../../') # カレントディレクトリを一階層上へ
print(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import synchronization.community_creater as community_creater

# 自作メソッド
import cows.geography as geography
import synchronization.functions.utility as my_utility
from synchronization.detection_model.cut_point_search import cut_point_search, cut_point_search_score

"""
recordにある牛の行動を可視化する
"""
def load_csv(filename):
    """ CSVファイルをロードし，記録内容をリストで返す """
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        records = [row for row in reader]
    return records

def extract(datalist, start, end):
    """ 行動分類リストから所定の時刻を取り出す """
    time_idx = []
    behavior_list = []
    for row in datalist:
        time = datetime.datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")
        if (start <= time and time < end):
            time_idx.append(time)
            behavior_list.append(int(row[3]))
    return time_idx, behavior_list

def plot_bar(time_idx, behavior_list, filename, change_list, change_point_scores, change_point_scores2, change_point_scores3):
    color_list = ['red', 'green', 'blue', 'orange', 'purple']
    fig = plt.figure(figsize=(19.2, 9.67))
    plt.subplots_adjust(left=0.10, right=0.90, bottom=0.05, top=0.95, hspace=0.35)
    ax = fig.add_subplot(4, 1, 1)
    _plot_color_bar(ax, time_idx, behavior_list, "behavior (red: resting, green: grazing, blue: walking)", color_list)
    ax3 = fig.add_subplot(4, 1, 2)
    ax3.set_ylim((0, 650))
    ax3.plot(time_idx, change_point_scores)
    ax4 = fig.add_subplot(4, 1, 3)
    ax4.set_ylim((0, 300))
    ax4.plot(time_idx, change_point_scores2)
    ax5 = fig.add_subplot(4, 1, 4)
    ax5.set_ylim((0, 100))
    ax5.plot(time_idx, change_point_scores3)
    # ax2 = fig.add_subplot(3, 1, 3)
    # _plot_color_bar(ax2, time_idx, change_list, "change point flag", ["white", "green"])
    pdb.set_trace()
    plt.savefig(filename)
    plt.close()
    print(filename + "を作成しました")
    return

def _plot_color_bar(ax, idx_list, series, name, color_list):
    ax.set_title(name)
    for i, data in enumerate(series):
        if (i < len(series) - 1):
            left = idx_list[i]
            right = idx_list[i+1]
            ax.axvspan(left, right, color=color_list[int(data)], alpha=0.8)
    return

if __name__ == "__main__":
    record_file = "./synchronization/detection_model/record.csv"
    output_file = "./synchronization/detection_model/"
    behavior_file = "./behavior_information/"
    records = load_csv(record_file)
    for i, row in enumerate(records):
        if (int(row[5]) == 1):
            # 牛の行動をファイルからロード
            target_cow_id = row[4][:5]
            date = datetime.datetime.strptime(row[0] + " " + row[1], "%Y/%m/%d %H:%M:%S") - datetime.timedelta(hours=9) # 翌朝の可能性もあるので時間を9時間戻す
            behaviors = load_csv(behavior_file + date.strftime("%Y%m%d/") + str(target_cow_id) + ".csv")
            # 変化点部分の行動を切り取る
            start = datetime.datetime.strptime(row[0] + " " + row[1], "%Y/%m/%d %H:%M:%S")
            end = datetime.datetime.strptime(row[2] + " " + row[3], "%Y/%m/%d %H:%M:%S") + datetime.timedelta(seconds=5) # 不等号で抽出するため5秒追加
            time_idx, behaviors = extract(behaviors[1:], start, end)
            
            # 行動分岐点を探す
            change_point_series, change_point_scores = cut_point_search_score(behaviors)

            # プロット
            plot_bar(time_idx, behaviors, output_file + str(target_cow_id) + ".jpg", change_point_series, change_point_scores[0][0], change_point_scores[1][0]+change_point_scores[2][0], change_point_scores[3][0] + change_point_scores[4][0] + [0 for i in range(len(change_point_scores[2][0]))])
