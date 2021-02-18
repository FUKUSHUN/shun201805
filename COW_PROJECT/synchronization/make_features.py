import os, sys
import time
import datetime
import numpy as np
import pandas as pd
import pdb
import pickle

# 自作クラス
os.chdir('../') # カレントディレクトリを一階層上へ
print(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import synchronization.community_creater as community_creater
import synchronization.interaction_analyzer as interaction_analyzer
import synchronization.functions.utility as my_utility
from synchronization.graph_operation.graph_series import GraphSeriesAnalysis
from synchronization.set_operation.set_series import SetSeriesAnalysis
from synchronization.topic_model.lda import GaussianLDA

# 自作ライブラリ
import synchronization.topic_model.make_session as make_session
import synchronization.topic_model.session_io as session_io

delta_c = 2 # コミュニティの抽出間隔 [minutes]
delta_s = 5 # データのスライス間隔 [seconds] 
epsilon = 10 # コミュニティ決定のパラメータ
dzeta = 12 # コミュニティ決定のパラメータ
leng = 1 # コミュニティ決定のパラメータ
start = datetime.datetime(2018, 5, 1, 0, 0, 0)
end = datetime.datetime(2018, 7, 31, 0, 0, 0)
target_list = ['20122','20129','20158','20170','20192','20197','20215','20267','20283'] # 2018/5/1 - 2018/7/31
# target_list = ['20113', '20118', '20126', '20170', '20255', '20295', '20299'] # 2018/9/10 - 2018/12/25
# target_list = ['20115', '20117', '20127', '20131', '20171', '20220', '20256', '20283', '20303'] # 2019/3/20 - 2019/7/3
cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
change_point_file = "./synchronization/estrus_detection/"

def make_features():
    """ 特徴を作成し各セッションを分析する """
    global start, end
    date = start
    while (date < end):
        s1 = time.time()
        communities_list = []
        community_graph_list = []
        interaction_graph_list = []
        t_list = []
        cow_id_list = my_utility.get_existing_cow_list(date, cows_record_file)
        com_creater = community_creater.CommunityCreater(date, cow_id_list)
        cow_id_list = com_creater.cow_id_list
        # --- 行動同期を計測する ---
        t = date + datetime.timedelta(hours=12) # 正午12時を始まりとするが.......ときに9時始まりのときもある
        t_start = date + datetime.timedelta(hours=12) # 正午12時を始まりとする
        t_end = date + datetime.timedelta(days=1) + datetime.timedelta(hours=9) # 翌午前9時を終わりとする
        while (t < t_end):
            t_list.append(t)
            interaction_graph = com_creater.make_interaction_graph(t, t+datetime.timedelta(minutes=delta_c), method="position", delta=delta_s, epsilon=epsilon, dzeta=dzeta) \
                if (t_start <= t) else np.array([[]]) # 重み付きグラフを作成
            community = com_creater.create_community(t, t+datetime.timedelta(minutes=delta_c), interaction_graph, delta=delta_s, leng=leng) \
                if (t_start <= t) else [[]] # コミュニティを決定
            com_creater.visualize_position(t, t+datetime.timedelta(minutes=delta_c), community, target_cow_id='20170', delta=delta_s) # 位置情報とコミュニティをプロット1
            community_graph = com_creater.get_community_graph(community)
            interaction_graph_list.append(interaction_graph)
            communities_list.append(community)
            community_graph_list.append(community_graph)
            t += datetime.timedelta(minutes=delta_c)
        e1 = time.time()
        print("処理時間", (e1-s1)/60, "[min]")
        # --- 変化点を検知し，セッションを作る ---
        s2 = time.time()
        behavior_synch = com_creater.get_behavior_synch()
        position_synch = com_creater.get_position_synch()
        graph_analyzer = GraphSeriesAnalysis(cow_id_list, interaction_graph_list, "Poisson")
        for cow_id in target_list:
            if (cow_id in cow_id_list):
                # graph_analyzer.visualize_graph(cow_id, t_list) # グラフをまとめて可視化
                change_points, score_list = graph_analyzer.detect_change_point(cow_id, 5, 5, threshold=400) # 変化点検知
                # df = pd.concat([pd.Series(t_list), pd.Series(score_list), pd.Series(change_points)], axis=1, names=["time", "score", "change_flag"])
                # df.to_csv("./synchronization/graph_operation/"+ str(cow_id) + ".csv") # csvで保存
                community_list = make_session.get_focused_community(communities_list, cow_id) # セッションを作成するために対象牛の所属するコミュニティを抽出
                inte_analyzer = interaction_analyzer.InteractionAnalyzer(cow_id, behavior_synch, position_synch) # 特徴量を作成するクラス
                start_end_list = _get_start_end(t_list, change_points)
                feature_list = []
                for (start_point, end_point) in start_end_list:
                    try:
                        community_series, graph_series = _extract_community(t_list, community_list, interaction_graph_list, start_point, end_point)
                        features = inte_analyzer.extract_feature(start_point, end_point, community_series, delta_c=delta_c)
                        ave_dence = calculate_average_graph_density(community_series, graph_series, cow_id_list)
                        ave_iso = calculate_average_graph_isolation(community_series, graph_series, cow_id_list)
                        max_cow, same_time = look_for_cow_having_max_weight(community_series, graph_series, cow_id_list, cow_id)
                        # 開始時刻, 平均密度, 平均孤立度, 非休息割合, セッション長, 同じコミュニティになった回数, その牛
                        feature_list.append([start_point, end_point, ave_dence, ave_iso, 1 - features[2], features[0], same_time, max_cow])
                    except KeyError:
                        pdb.set_trace()
                my_utility.write_values(change_point_file + str(cow_id) + ".csv", feature_list)
            else:
                my_utility.write_values(change_point_file + str(cow_id) + ".csv", [[t_start, t_end]])
        e2 = time.time()
        print("処理時間", (e2-s2)/60, "[min]")
        date += datetime.timedelta(days=1)
    return

def _get_start_end(time_list, change_points):
    """ 変化点の始点と終点の時刻を格納したリストを作る (最後の時刻を強制的に終了時刻としている)
        Parameter
        time_list: 時刻のリスト
        change_points:  変化点なら1，そうでなければ0が格納されたリスト
        Return
        ret_list:   (start_time, end_time)を格納したリスト """
    ret_list = []
    is_first = True
    is_end = False
    i = 0
    for time, is_changed in zip(time_list, change_points):
        i += 1
        if (is_first): # 一番最初のみ開始時刻を設定
            start = time
            is_first = False
        if (is_changed == 1): # 変化点のとき
            end = time
            ret_list.append((start, end)) # 開始と終了の時刻をタプルで追加
            start = time # 次のセグメントの開始時刻を設定
        elif (i == len(time_list)): # 最後のタイムスタンプのとき
            end = time
            ret_list.append((start, end)) # 最後のセグメントを追加
    return ret_list

def _extract_community(time_list, community_list, graph_list, start, end):
    """ 当該時刻間のコミュニティとグラフをリストから抽出する """
    community_list_ret = []
    graph_list_ret = []
    is_start = False
    is_end = False
    for time, community, graph in zip(time_list, community_list, graph_list):
        if (time == start):
            is_start = True
        if (time == end):
            is_end = True
        if (is_start and not is_end):
            community_list_ret.append(community)
            graph_list_ret.append(graph)
    return community_list_ret, graph_list_ret

def calculate_average_graph_density(community_series, graph_series, cow_id_list):
    """ ある牛の所属するコミュニティのグラフ密度の平均を算出する """
    density_list = []
    for community, graph in zip(community_series, graph_series):
        # targetの所属するコミュニティのメンバのみからなるグラフに成形する
        index_list = [cow_id_list.index(str(cow_id)) for cow_id in community] # インタラクショングラフの何行目かのインデックス
        index_list = sorted(index_list)
        formed_graph = _form_graph(graph, index_list)
        # グラフの密度を算出する
        density = _calculate_graph_density(formed_graph)
        density_list.append(density)
    return sum(density_list) / len(density_list)

def _form_graph(W, index_list):
    """ 全体のグラフ（行列）からインデックス行（列）のみを取り出して成形する """
    graph_rows = [W[index] for index in index_list]
    formed_graph = np.stack(graph_rows, axis=0)
    graph_columns = [formed_graph[:,index] for index in index_list]
    formed_graph = np.stack(graph_columns, axis=1)
    return formed_graph

def _calculate_graph_density(W):
    """ グラフの密度を求める
        W:  コミュニティメンバのみに成形後のインタラクショングラフ """
    # 辺の数（重みが0より大きい）を数え上げる
    K = len(W)
    count = 0
    for i in range(K):
        for j in range(K):
            if (i < j and 0 < W[i,j]):
                count += 1
    # 完全グラフの時の辺の数で割り密度を算出する
    density = count / (K * (K-1) / 2) if 1 < K else 0
    return density

def calculate_average_graph_isolation(community_series, graph_series, cow_id_list):
    """ コミュニティのグラフ全体に対する孤立度を求める """
    isolation_list = []
    for community, graph in zip(community_series, graph_series):
        index_list = [cow_id_list.index(str(cow_id)) for cow_id in community] # インタラクショングラフの何行目かのインデックス
        index_list = sorted(index_list)
        formed_graph_out = _form_graph2(graph, index_list)
        M = np.sum(formed_graph_out) / 2
        formed_graph_in = _form_graph(graph, index_list)
        m = np.sum(formed_graph_in) / 2
        if (M != 0):
            iso = m / M
        else:
            iso = 1 # 自分しかコミュニティにいない場合M=0となる．この場合，孤立度は1 (最大)
        isolation_list.append(iso)
    return sum(isolation_list) / len(isolation_list)

def _form_graph2(graph, index_list):
    """ index_listにある行列のみ値を保持したグラフを生成する """
    N = len(graph)
    ret_graph = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if (i in index_list or j in index_list):
                ret_graph[i,j] = graph[i,j]
    return ret_graph

def look_for_cow_having_max_weight(community_series, graph_series, cow_id_list, target_cow_id):
    """ グラフのエッジの重みの平均が最大となる牛の個体番号ならびにセッション内で同じコミュニティだった時間を算出する """
    N = len(graph_series[0])
    sum_graph = np.zeros((N, N)) # すべてのグラフの和
    # グラフ系列（隣接行列）をすべて足し合わせる
    for graph in graph_series:
        sum_graph += graph
    target_index = cow_id_list.index(target_cow_id)
    max_cow_index = np.where(sum_graph[target_index] == max(sum_graph[target_index]))[0][0] # グラフのエッジの重みが対象牛に対して最大だったノードのインデックスを取得
    max_cow = cow_id_list[max_cow_index] # 牛の個体番号を求める
    # その牛と同じコミュニティであった回数を取得する
    count = 0
    for community in community_series:
        if (max_cow in community):
            count += 1
    return max_cow, count

if __name__ == '__main__':
    make_features()
