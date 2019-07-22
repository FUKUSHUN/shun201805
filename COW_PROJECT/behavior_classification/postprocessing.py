"""
このコードは圧縮後のデータ復元に必要な処理をまとめたものである
特徴出力後のcsvファイルからの復元時に活用する
"""

import sys

"""
分類結果のリストを時間のリストに合う形に整形する
Parameter
	results	: 分類結果のリスト
"""
def make_labels(results):
	new_results = []
	for r in results:
		new_results.append(r)
		new_results.append(r)
	return new_results

"""
特徴抽出したCSVファイル (圧縮済み) から元の時系列データを作成 (解凍) する
Parameter
    t_list    : 元の時系列データ
	zipped_t_list	: 圧縮後の時系列データ
	zipped_l_list	: 圧縮データから得られるラベルのデータ
Return
    new_t_list  : 新しい時間のリスト（削除部分との整合性をとるため）
    l_list  : ラベルのリスト
圧縮後の時系列データは (start-end) を休息 | 歩行の形で記載している
例) 2018/12/30 13:00:30-2018/12/30 13:00:30 | 2018/12/30 13:00:35-2018/12/30 13:00:45
"""
def decompress(t_list, zipped_t_list, zipped_l_list):
	print(sys._getframe().f_code.co_name, "実行中")
	index = 0
	new_t_list = []
	l_list = []
	start = zipped_t_list[0][0]
	end = zipped_t_list[0][1]
	label = zipped_l_list[0]
	final = zipped_t_list[len(zipped_t_list)-1][1]
	for time in t_list:
		if (start <= time and time <= end):
			new_t_list.append(time)
			l_list.append(label)
		if (final <= time):
			break
		if (end <= time):
			index += 1
			start = zipped_t_list[index][0]
			end = zipped_t_list[index][1]
			label = zipped_l_list[index]
	print(sys._getframe().f_code.co_name, "正常終了\n")
	return new_t_list, l_list

"""
古い時間のリストのうち，新しい時間のリストに含まれている部分のみを切り取る
Parameter
    old_t_list  : 古い時間のリスト
    new_t_list  : 新しい時間のリスト
    something   : 何かのリスト（古い時間のリストと一対一対応）
"""
def make_new_list(old_t_list, new_t_list, something):
    index = 0
    new_something = []
    start = new_t_list[0]
    end = new_t_list[len(new_t_list) - 1]
    for time in old_t_list:
        if (start <= time and time <= end):
            new_something.append(something[index])
            index += 1
    return new_something