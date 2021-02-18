import os, sys
import time
import datetime
import numpy as np
import pandas as pd
import pdb
import pickle

# 自作クラス
os.chdir('../../') # カレントディレクトリを一階層上へ
print(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import synchronization.community_creater as community_creater
import synchronization.interaction_analyzer as interaction_analyzer
import synchronization.functions.utility as my_utility
import cows.geography as geography

def get_score_matrix(beh_df):
    """ スコア行列を獲得する """
    score_matrix = np.zeros((3, 3))
    count_vector = np.zeros(3)
    for i in range(3):
        count_vector[i] = (beh_df == i).values.sum()
    for i in range(3):
        score_matrix[i, i] = (count_vector[(i+1)%3] + count_vector[(i+2)%3]) / (count_vector.sum() * 2)
    return score_matrix

def score_synchro(beh_df, pos_df, target_cow_id, community, score_matrix, dis_threshold=10):
    """ 同期をスコア化する """
    score_dict = {} # 返却値
    for c in community:
        score_dict[c] = 0 # 牛のIDをキーにスコアを格納する
    target_beh = beh_df[str(target_cow_id)].values
    target_pos = pos_df[str(target_cow_id)].values
    community2 = community.copy()
    if (target_cow_id in community2):
        community2.remove(target_cow_id)
    for cow_id in community2:
        score = 0
        nearcow_pos = pos_df[cow_id].values
        nearcow_beh = beh_df[cow_id].values
        for i in range(len(target_beh)):
            lat1, lon1 = target_pos[i][0], target_pos[i][1]
            lat2, lon2 = nearcow_pos[i][0], nearcow_pos[i][1]
            dis, _ = geography.get_distance_and_direction(lat1, lon1, lat2, lon2, True)
            # 近い距離にいれば同期しているかを確認する
            if (dis <= dis_threshold and _check_position(lat1, lon1) and _check_position(lat2, lon2)):
                score += score_matrix[target_beh[i], nearcow_beh[i]]
        score_dict[cow_id] = score / 12 # 1分間あたりになおす
    return score_dict

def _check_position (lat, lon):
    """ 牛舎や屋根下にいないかチェック """
    cowshed_boundary = (34.882449, 134.863557) # これより南は牛舎
    roof_boundary = [(34.882911, 134.863357), (34.882978, 134.863441)] # 肥育横の屋根の座標（南西，北東）
    if (lat < cowshed_boundary[0]):
        return False
    if (roof_boundary[0][0] < lat and lat < roof_boundary[1][0] and roof_boundary[0][1] < lon and lon < roof_boundary[1][1]):
        return False
    return True

if __name__ == "__main__":
    cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
    dir_name = "./synchronization/detection_model/test/"
    target_cow_id = '20192'
    delta_s = 5 # データのスライス間隔 [seconds]
    epsilon = 10 # コミュニティ決定のパラメータ
    leng = 1 # コミュニティ決定のパラメータ
    date_list = [
        datetime.datetime(2018, 5, 14, 0, 0, 0),
        datetime.datetime(2018, 5, 20, 0, 0, 0),
        datetime.datetime(2018, 5, 27, 0, 0, 0), 
        datetime.datetime(2018, 6, 14, 0, 0, 0),
        datetime.datetime(2018, 6, 17, 0, 0, 0)
    ]
    for date in date_list:
        start = date + datetime.timedelta(hours=12)
        end = date + datetime.timedelta(days=1) + datetime.timedelta(hours=9)
        # 牛のリストを入手
        cow_id_list = my_utility.get_existing_cow_list(date, cows_record_file)
        com_creater = community_creater.CommunityCreater(date, cow_id_list)
        cow_id_list = com_creater.cow_id_list
        interaction_graph = com_creater.make_interaction_graph(start, end, method='behavior', delta=delta_s, epsilon=epsilon)
        community = com_creater.create_community(start, end, interaction_graph, delta=delta_s, leng=leng)
        community = [com for com in community if str(target_cow_id) in com][0]

        # 行動と位置のリストを入手
        behavior_synch = com_creater.get_behavior_synch()
        position_synch = com_creater.get_position_synch()
        behaviors = behavior_synch.extract_df(start, end, delta_s) [community]
        positions = position_synch.extract_df(start, end, delta_s) [community]

        # 全頭の同期度を測定
        value_list = [community]
        score_matrix = get_score_matrix(behaviors)
        for cow_id in community:
            score_dict = score_synchro(behaviors, positions, cow_id, community, score_matrix)
            value_list.append([value for value in score_dict.values()])
        my_utility.write_values(dir_name + date.strftime("%Y%m%d") + ".csv", value_list)
