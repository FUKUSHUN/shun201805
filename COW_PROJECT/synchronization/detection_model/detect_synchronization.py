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
import behavior_classification.models.beysian.mixed_model as mixed_model
from synchronization.graph_operation.graph_series import GraphSeriesAnalysis
from synchronization.functions.plotting import PlotMaker2D

# 自作メソッド
import cows.geography as geography
import synchronization.functions.utility as my_utility
from synchronization.detection_model.cut_point_search import cut_point_search, estimate_parameters

"""
recordにある牛の行動を可視化する
"""
def load_csv(filename):
    """ CSVファイルをロードし，記録内容をリストで返す """
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        records = [row for row in reader]
    return records

def fetch_information(date):
    """ 1日分の牛のデータを読み込む """
    date = datetime.datetime(date.year, date.month, date.day, 0, 0, 0)
    cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
    cow_id_list = my_utility.get_existing_cow_list(date, cows_record_file)
    com_creater = community_creater.CommunityCreater(date, cow_id_list)
    cow_id_list = com_creater.cow_id_list
    return cow_id_list, com_creater

def cut_data(behaviors, positions, change_point_series):
    """ 変化点に従って，データを分割し，リスト化して返す """
    df_list1 = [] # behavior
    df_list2 = [] # position
    change_idx = [i for i, flag in enumerate(change_point_series) if flag == 1]
    before_idx = change_idx[0]
    for idx in range(1, len(change_idx)):
        df_list1.append(behaviors[before_idx: change_idx[idx]])
        df_list2.append(positions[before_idx: change_idx[idx]])
        before_idx = change_idx[idx]
    df_list1.append(behaviors[change_idx[-1]:]) # 最後の行動分岐点から最後まで
    df_list2.append(positions[change_idx[-1]:]) # 最後の行動分岐点から最後まで
    return df_list1, df_list2

def get_change_time(time_series, change_point_series):
    """ 変化点に従って, (start, end) のリストを作る """
    start_end_list = []
    change_idx = [i for i, flag in enumerate(change_point_series) if flag == 1]
    before_idx = 0
    for i in range(1, len(change_idx)):
        start = time_series[before_idx]
        end = time_series[change_idx[i]]
        start_end_list.append((start, end))
        before_idx = change_idx[i]
    return start_end_list

def plot_bar(time_idx, behavior_list, filename, change_list, session_num_list):
    fig = plt.figure(figsize=(19.2, 9.67))
    plt.subplots_adjust(left=0.10, right=0.90, bottom=0.05, top=0.95, hspace=0.35)
    ax = fig.add_subplot(3, 1, 1)
    _plot_color_bar(ax, time_idx, behavior_list, "behavior (red: resting, green: grazing, blue: walking)", ['red', 'green', 'blue'])
    ax2 = fig.add_subplot(3, 1, 2)
    _plot_color_bar(ax2, time_idx, change_list, "change point flag", ["white", "green"])
    ax3 = fig.add_subplot(3, 1, 3)
    _plot_color_bar(ax3, time_idx, session_num_list, "session number", ['orange', 'purple', 'white', 'yellow', 'black'])
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
    delta_s = 5 # データのスライス間隔 [seconds]
    delta_c = 2 # コミュニティの生成間隔 [minutes]
    epsilon = 10 # コミュニティ決定のパラメータ
    dzeta = 12 # コミュニティ決定のパラメータ
    leng = 1 # コミュニティ決定のパラメータ
    record_file = "./synchronization/detection_model/record.csv"
    output_file = "./synchronization/detection_model/"
    records = load_csv(record_file)
    segment_parameter_list = []
    for i, row in enumerate(records):
    
        # レコードファイルから日付と牛のIDを入手
        target_cow_id = row[4][:5]
        date = datetime.datetime.strptime(row[0] + " " + row[1], "%Y/%m/%d %H:%M:%S") - datetime.timedelta(hours=9) # 翌朝の可能性もあるので時間を9時間戻す
        date = datetime.datetime(date.year, date.month, date.day, 0, 0, 0)
        cow_id_list, com_creater = fetch_information(date)
        
        # インタラクショングラフを作成する
        interaction_graph_list = []
        t_list = []
        t = date + datetime.timedelta(hours=12) # 正午12時を始まりとするが.......ときに9時始まりのときもある
        t_start = date + datetime.timedelta(hours=12) # 正午12時を始まりとする
        t_end = date + datetime.timedelta(days=1) + datetime.timedelta(hours=9) # 翌午前9時を終わりとする
        while (t < t_end):
            print(t.strftime("%Y/%m/%d %H:%M:%S"))
            t_list.append(t)
            interaction_graph = com_creater.make_interaction_graph(t, t+datetime.timedelta(minutes=delta_c), method="position", delta=delta_s, epsilon=epsilon, dzeta=dzeta) \
                if (t_start <= t) else np.array([[]]) # 重み付きグラフを作成
            interaction_graph_list.append(interaction_graph)
            t += datetime.timedelta(minutes=delta_c)

        # コミュニティ構造の変化点検知を行う
        graph_analyzer = GraphSeriesAnalysis(cow_id_list, interaction_graph_list, "Poisson")
        change_points, _ = graph_analyzer.detect_change_point(str(target_cow_id), 5, 5, threshold=400) # 変化点検知
        session_times = get_change_time(t_list, change_points)
        for t_idx in session_times:
            start, end = t_idx[0], t_idx[1]
            interaction_graph = com_creater.make_interaction_graph(start, end, method="position", delta=delta_s, epsilon=epsilon, dzeta=dzeta)
            community = com_creater.create_community(start, end, interaction_graph, delta=delta_s, leng=leng)
            community = [com for com in community if str(target_cow_id) in com][0]

            # 行動分岐点を探す
            behaviors = com_creater.get_behavior_synch().extract_df(start, end, delta_s)
            positions = com_creater.get_position_synch().extract_df(start, end, delta_s)
            behaviors = behaviors[community]
            positions = positions[community]
            change_point_series = cut_point_search(behaviors[str(target_cow_id)].values.tolist())
            b_segments, p_segments = cut_data(behaviors, positions, change_point_series)
            for b_seg, p_seg in zip(b_segments, p_segments):
                theta = estimate_parameters(b_seg[str(target_cow_id)])
                segment_parameter_list.append(len(b_seg[str(target_cow_id)]) * theta)
    
    segment_parameter_list = [par[:3] for par in segment_parameter_list]
    my_utility.write_values("./synchronization/detection_model/segment.csv", segment_parameter_list)
    cov_matrixes = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), \
                        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), \
                            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), \
                                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), \
                                    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])]
    mu_vectors = [np.array([30, 1, 1]), np.array([15, 10, 1]), np.array([8, 10, 20]), np.array([2, 8, 8]), np.array([20, 20, 20])]
    pi_vector = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
    alpha_vector = np.array([1, 1, 1, 1, 1])
    max_iterater = 500
    mixture_model = mixed_model.GaussianMixedModel(cov_matrixes, mu_vectors, pi_vector, alpha_vector, max_iterater)
    mixture_model.gibbs_sample(segment_parameter_list, np.array([[5, 5, 5]]).T, 1, 5, np.eye(3))
    
    # x = [x[0] for x in segment_parameter_list]
    # y = [x[1] for x in segment_parameter_list]
    # plt.scatter(x, y)
    # mu, cov = mixture_model.get_gaussian_parameters()
    # plt.scatter([m[0] for m in mu], [m[1] for m in mu], c="red")
    
    for i, row in enumerate(records):
        # 牛の行動をファイルからロード
        target_cow_id = row[4][:5]
        date = datetime.datetime.strptime(row[0] + " " + row[1], "%Y/%m/%d %H:%M:%S") - datetime.timedelta(hours=9) # 翌朝の可能性もあるので時間を9時間戻す
        date = datetime.datetime(date.year, date.month, date.day, 0, 0, 0)
        cow_id_list, com_creater = fetch_information(date)
        # 変化点部分の行動を切り取る
        start = datetime.datetime.strptime(row[0] + " " + row[1], "%Y/%m/%d %H:%M:%S")
        end = datetime.datetime.strptime(row[2] + " " + row[3], "%Y/%m/%d %H:%M:%S") + datetime.timedelta(seconds=5) # 不等号で抽出するため5秒追加
        behaviors = com_creater.get_behavior_synch().extract_df(start, end, delta_s)
        positions = com_creater.get_position_synch().extract_df(start, end, delta_s)
        
        # 行動分岐点を探す
        change_point_series = cut_point_search(behaviors [str(target_cow_id)])
        b_segments, p_segments = cut_data(behaviors, positions, change_point_series)
        session_num_list = []
        for b_seg, p_seg in zip(b_segments, p_segments):
            theta = estimate_parameters(b_seg[str(target_cow_id)])
            prob, _ = mixture_model.predict([len(b_seg[str(target_cow_id)]) * theta])
            session_num = np.argmax(prob)
            session_num_list.extend((np.ones(len(b_seg[str(target_cow_id)])) * session_num).tolist())
        # プロット
        plot_bar([i for i in range(len(behaviors [str(target_cow_id)]))], behaviors [str(target_cow_id)], output_file + str(target_cow_id) + "_" + str(i) + ".jpg", change_point_series, session_num_list)
    pdb.set_trace()
