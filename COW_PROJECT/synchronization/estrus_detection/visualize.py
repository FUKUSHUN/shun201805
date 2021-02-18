import os, sys
import time
import datetime
import numpy as np
import pandas as pd
import pdb

# 自作クラス
os.chdir('../../') # カレントディレクトリを二階層上へ
print(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import synchronization.community_creater as community_creater
import synchronization.functions.utility as my_utility
import synchronization.functions.plotting as plotting

def extract_list(arrays, start, end):
    """ ある時間内のクエリを抽出する """
    extracted_query = []
    for row in arrays:
        time_stamp = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
        if (start <= time_stamp and time_stamp < end):
            extracted_query.append(row)
    return extracted_query

def score_one_part(arrays, method='sum'):
    """ 1つのパーティションごとにスコアを求める """
    score = -1 if (method == 'max') else 0
    for row in arrays:
        if (method == 'sum'):
            score += row[8] # scoreの列
        elif (method == 'max'):
            if (score < row[8]): # scoreの列
                score = row[8] # scoreの列
    return score

def search_most_stable_time(arrays):
    """ 1つのパーティションでもっともスコアが高かったセッションを求める """
    score = 0
    representitive = None
    for row in arrays:
        if (score < row[8]):
            score = row[8]
            representitive = row
    return representitive

def visualize_adjectory(dt, row, target_list, delta=5):
    s = time.time()
    tmp_time = dt - datetime.timedelta(hours=9) # 時差の分巻き戻しておく（翌朝になっている可能性もあるので）
    date = datetime.datetime(tmp_time.year, tmp_time.month, tmp_time.day, 0, 0, 0)
    cow_id_list = my_utility.get_existing_cow_list(date, cows_record_file)
    com_creater = community_creater.CommunityCreater(date, cow_id_list)
    for i, cow_data in enumerate(row): # 各牛ごとに見ていく
        if (cow_data is not None and cow_data is not np.nan):
            target_cow_id = target_list[i]
            try:
                opponent_cow_id = str(int(cow_data[7]))
            except:
                pdb.set_trace()
            start_t = datetime.datetime.strptime(cow_data[0], '%Y-%m-%d %H:%M:%S')
            end_t = datetime.datetime.strptime(cow_data[1], '%Y-%m-%d %H:%M:%S')
            com_creater.visualize_adjectory(start_t, end_t, [target_cow_id, opponent_cow_id], target_cow_id=target_cow_id, delta=delta) # 軌跡をプロット
            print("軌跡動画と画像を作成しました: ", target_cow_id, " -> ", start_t.strftime('%Y/%m/%d %H:%M:%S'))
    e = time.time()
    print("処理時間", (e-s)/60, "[min]")
    return

if __name__ == "__main__":
    cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
    dir_name = './synchronization/estrus_detection/'
    # target_list = ['20122','20129','20158','20170','20192','20197','20215','20267','20283'] # 2018/5/1 - 2018/7/31
    target_list = ['20113', '20118', '20126', '20170', '20255', '20295', '20299'] # 2018/9/10 - 2018/12/25
    # target_list = ['20115', '20117', '20127', '20131', '20171', '20220', '20256', '20283', '20303'] # 2019/3/20 - 2019/7/3
    visualize_start = datetime.datetime(2019, 3, 20, 0, 0, 0) # 軌跡画像の描画を開始する日
    visualize_end = datetime.datetime(2019, 6, 20, 0, 0, 0) # 軌跡画像の描画を終了する日
    score_df = pd.DataFrame([])
    for target_cow_id in target_list:
        filename = dir_name + target_cow_id + '.csv'
        # csvファイルをロードする
        df = pd.read_csv(filename, engine='python', usecols=[0, 1, 2, 3, 4, 5, 6, 7], names=['start_time', 'end_time', 'dence', 'isolation', 'non-rest', 'interval', 'max_time', 'opponent_cow'])
        df.fillna(0, inplace=True) # Noneを0で埋める
        df['score'] = df['non-rest'] * df['dence'] * df['max_time']
        # パーティション (1日，半日など) に分けてリストを組む
        partition_list = []
        score_list = df.values.tolist() # リストに変換
        beginning_time = datetime.datetime.strptime(score_list[0][0], '%Y-%m-%d %H:%M:%S') # ファイル全体の開始時刻
        ending_time = datetime.datetime.strptime(score_list[len(score_list)-1][1], '%Y-%m-%d %H:%M:%S') # ファイル全体の終了時刻
        date_start = beginning_time
        time_series = []
        while (date_start < ending_time):
            date_end = date_start + datetime.timedelta(hours=21)
            start = date_start
            while (start < date_end): # 1日の範囲で
                time_series.append(start)
                end = start + datetime.timedelta(hours=21)
                partition = extract_list(score_list, start, end)
                partition_list.append(partition) # 期間内のデータを一つのパーティションとしてリストに登録
                start = end
            date_end += datetime.timedelta(hours=3)
            date_start = date_end
        # 1パーティションごとにスコアを算出する
        partition_score_list = []
        most_stable_list = []
        for partition in partition_list:
            partition_score = score_one_part(partition, method='sum')
            partition_score_list.append(partition_score)
            most_stable_list.append(search_most_stable_time(partition))
        plot_maker = plotting.PlotMaker2D(xmin=None, xmax=None, ymin=None, ymax=None, title='estrus scores', xlabel='time_series', ylabel='score', figsize=(19.2, 10.8))
        plot_maker.adjust_margin(top=0.95, bottom=0.15, right=0.95, left=0.05)
        plot_maker.create_bar_graph(time_series, partition_score_list, width=0.4)
        plot_maker.save_fig(dir_name + 'figure/' + target_cow_id + ".jpg")
        # 最も安定的だった期間の軌跡描画を行うためのデータフレームを作成
        tmp_df = pd.concat([pd.Series(time_series, name='Time'), pd.Series(most_stable_list, name=target_cow_id)], axis=1)
        tmp_df = tmp_df.set_index('Time')
        score_df = pd.concat([score_df, tmp_df], axis=1)
    # # 軌跡描画を行う
    # for dt, row in score_df.iterrows():
    #     if (visualize_start <= dt and dt < visualize_end):
    #         visualize_adjectory(dt, row, target_list)