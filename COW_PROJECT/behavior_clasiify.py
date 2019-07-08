#-*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
import statistics
import csv
import re
import datetime
import gc
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import sklearn.decomposition as skd
import analyze_main.hmm as hmm

import cows.cow as Cow
import cows.geography as geo
import cows.momentum_analysys as ma
import image.adjectory_image as disp
"""
えげつないスパゲッティです
"""
#データベースから指定牛の指定期間のデータを読み込む (元データ (1 Hz) を5s (0.2Hz) に戻す処理も含む)
def read_gps(cow_id, start, end):
	# データを読み込み，それぞれのリストを作成 (情報ごとのリストにするか時間ごとのリストのリストにするかは場合による) 
	time_list = [] # 時間のリスト (主キー)
	position_list = [] # (緯度，経度) のリスト
	distance_list = [] # 距離のリスト
	velocity_list = [] # 速さのリスト
	angle_list = [] # 移動角度のリスト
	dt = datetime.datetime(start.year, start.month, start.day)
	a = start
	while(dt < end):
		t_list = []
		pos_list = []
		dis_list = []
		vel_list = []
		ang_list = []
		cow = Cow.Cow(cow_id, dt)
		dt = dt + datetime.timedelta(days = 1)
		while(a <= dt and a < end):
			gps_list = cow.get_gps_list(a, a + datetime.timedelta(minutes = 60))
			g_before = None
			for i in range(int(len(gps_list) / 5)):
				g = gps_list[i * 5]
				if g_before is not None:
					lat1, lon1, vel1 = g_before.get_gps_info(g_before.get_datetime())
					lat2, lon2, vel2 = g.get_gps_info(g.get_datetime())
					distance, angle = geo.get_distance_and_direction(lat1, lon1, lat2, lon2, False)
					#print(g.get_datetime().strftime("%Y/%m/%d %H:%M:%S") + " : ", lat2 , ",", lon2)
					t_list.append(g.get_datetime()) #時間の格納
					pos_list.append(geo.translate(lat2, lon2)) #位置情報の格納
					dis_list.append(distance) #距離の格納
					vel_list.append(vel2) #速さの格納
					ang_list.append(angle) #角度の格納
				g_before = g
			a = a + datetime.timedelta(minutes = 60)
			del gps_list
			gc.collect()
		time_list.append(t_list) #1日分の時間のリストの格納
		position_list.append(pos_list) #1日分の位置情報の格納
		distance_list.append(dis_list) #1日分の距離のリストの格納
		velocity_list.append(vel_list) #1日分の速さのリストの格納
		angle_list.append(ang_list) #1日分の角度のリストの格納
		del cow
		gc.collect()
		a = dt
	return time_list, position_list, distance_list, velocity_list, angle_list

#データの解析に使用する時間 (午後12時-午前9時 JST) 分を抽出する
def select_use_time(t_list, p_list, d_list, v_list, a_list):
	knot = 0.514444 # 1 knot = 0.51444 m/s
	time_tmp_list = []
	position_tmp_list = []
	distance_tmp_list = []
	velocity_tmp_list = []
	angle_tmp_list = []
	for (t, p, d, v, a) in zip(t_list, p_list, d_list, v_list, a_list):
		t = t + datetime.timedelta(hours = 9)
		if(t.hour < 9 or 12 < t.hour):
			time_tmp_list.append(t)
			position_tmp_list.append(p)
			distance_tmp_list.append(d) 
			velocity_tmp_list.append(v * knot) #単位を[m/s]に直しているだけ
			angle_tmp_list.append(a)
	return time_tmp_list, position_tmp_list, distance_tmp_list, velocity_tmp_list, angle_tmp_list

#プロットする
def plot_velocity_data(t_list, v_list):
	#グラフ化のための設定
	fig = plt.figure()
	ax1 = fig.add_subplot(1,1,1) #4行1列の図
	#ax1の設定
	ax1.set_xticklabels(t_list, rotation=90, fontsize='small') #ラベルを回転・サイズ指定
	ax1.xaxis.set_major_locator(mdates.AutoDateLocator()) #自動的にラベル表示の間隔を設定
	ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H-%M")) #日付の表示形式を決定
	ax1.plot(t_list, v_list, 'b')
	ax1.legend(("Velocity",), loc='upper left')
	#plt.savefig(t_list[0].strftime("%y-%m-%d") + "-test.png")
	plt.show()

#散布図形式でプロットする
def scatter_plot(t_list, v_list, c_list):
	x_list = []
	for i in range(len(t_list)):
		x_list.append(i)
	x = np.array(x_list)
	y = np.array(v_list)
	c = np.array(c_list)
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(x, y, c = c, s = 1)
	plt.show()

#圧縮操作を行う
"""
Parameter
	t_list	: 圧縮されていない時間のリスト
	p_list	: 圧縮されていない(緯度, 経度)のリスト
	d_list	: 圧縮されていない距離のリスト
	v_list	: 圧縮されていない速度のリスト
return
	zipped_list	: 各要素が (0: (start, end), 1: 重心 (緯度, 距離), 2: 総距離, 3: 平均速度, 4: 暫定的なラベル) の形のリスト
"""
def zipping(t_list, p_list, d_list, v_list, r_threshold = 0.0694, g_threshold = 0.181):
	print(sys._getframe().f_code.co_name, "実行中")
	print("圧縮を開始します---")
	zipped_list = []
	p_tmp_list = [] # 場所 (緯度, 軽度) 用
	d_tmp_list = [] # 距離用
	v_tmp_list = [] # 速度用
	is_rest = False
	is_graze = False
	is_walk = False

	start = None #未定義エラー除去
	end = None 
	for time, place, distance, velocity in zip(t_list, p_list, d_list, v_list):
		if (is_rest): # 1個前が休息
			if (choice_state(velocity) == 0):
				p_tmp_list.append(place)
				d_tmp_list.append(distance)
				v_tmp_list.append(velocity)
				end = time
			else: # 休息の終了
				p, d, v = extract_mean(p_tmp_list, d_tmp_list, v_tmp_list)
				zipped_list.append(((start, end), p, d, [v, max(v_tmp_list), min(v_tmp_list)], 0))
				p_tmp_list = [place]
				d_tmp_list = [distance]
				v_tmp_list = [velocity]
				start = time
				end = time
		elif (is_graze): # 1個前が採食
			if (choice_state(velocity) == 1):
				p_tmp_list.append(place)
				d_tmp_list.append(distance)
				v_tmp_list.append(velocity)
				end = time
			else: # 採食の終了
				p, d, v = extract_mean(p_tmp_list, d_tmp_list, v_tmp_list)
				zipped_list.append(((start, end), p, d, [v, max(v_tmp_list), min(v_tmp_list)], 1))
				p_tmp_list = [place]
				d_tmp_list = [distance]
				v_tmp_list = [velocity]
				start = time
				end = time
		elif (is_walk): # 1個前が歩行
			if (choice_state(velocity) == 2):
				p_tmp_list.append(place)
				d_tmp_list.append(distance)
				v_tmp_list.append(velocity)
				end = time
			else: # 歩行の終了
				p, d, v = extract_mean(p_tmp_list, d_tmp_list, v_tmp_list)
				zipped_list.append(((start, end), p, d, [v, max(v_tmp_list), min(v_tmp_list)], 2))
				p_tmp_list = [place]
				d_tmp_list = [distance]
				v_tmp_list = [velocity]
				start = time
				end = time
		else: # ループの一番最初だけここ
			start = time
			end = time
			p_tmp_list.append(place)
			d_tmp_list = [distance]
			v_tmp_list = [velocity]

		if (choice_state(velocity) == 0):
			is_rest = True
			is_graze = False
			is_walk = False
		elif (choice_state(velocity) == 1):
			is_rest = False
			is_graze = True
			is_walk = False
		else:
			is_rest = False
			is_graze = False
			is_walk = True
	
	# 最後の行動を登録して登録終了
	p, d, v = extract_mean(p_tmp_list, d_tmp_list, v_tmp_list)
	zipped_list.append(((start, end), p, d, [v, max(v_tmp_list), min(v_tmp_list)], choice_state(v)))
	print("---圧縮が終了しました")
	print(sys._getframe().f_code.co_name, "正常終了\n")
	return zipped_list

#休息・採食・歩行を判断する（今は閾値や最近傍のアプローチだが変更する可能性あり）
def choice_state(velocity, r_threshold = 0.0694, g_threshold = 0.181):
	if (velocity < r_threshold):
		return 0 # 休息
	elif (r_threshold <= velocity and velocity < g_threshold):
		return 1 # 採食
	else:
		return 1 # 歩行 (実施不透明)

#場所，距離，速さに関してリスト内のそれぞれ重心，総和，平均を求める
def extract_mean(p_list, d_list, v_list):
	ave_lat_list = [] #平均を求めるための緯度のリスト
	ave_lon_list = [] #平均を求めるための経度のリスト
	sum_dis_list = [] #休息中の距離の総和（おそらく休息中の移動距離の総和は0）を求めるための距離のリスト
	ave_vel_list = [] #休息中の速さの平均を求めるための速さのリスト
	for place, distance, velocity in zip(p_list, d_list, v_list):
		ave_lat_list.append(place[0])
		ave_lon_list.append(place[1])
		sum_dis_list.append(distance)
		ave_vel_list.append(velocity)
	
	lat = sum(ave_lat_list) / len(ave_lat_list)
	lon = sum(ave_lon_list) / len(ave_lon_list)
	dis = sum(sum_dis_list)
	vel = sum(ave_vel_list) / len(ave_vel_list)
	return (lat,lon), dis, vel

#休息時間と休息の重心（緯度・経度）を時刻順に格納したリストを返す
"""
Parameter
	zipped_list		: 圧縮後のリスト (各要素の１番目が (start, end), 2番目が (緯度, 経度))
return
	rest_list	: 休息の重心とその時間
"""
def make_rest_data(zipped_list):
	print(sys._getframe().f_code.co_name, "実行中")
	print("休息時間と休息の重心を時刻順にリストに格納します---")
	rest_list = []
	for row in zipped_list:
		if (row[4] == 0): #休息なら
			rest_list.append(((row[0][1] - row[0][0]).total_seconds() / 60, row[1][0], row[1][1])) # (時間 [minutes], lat, lon)
	print("---休息時間と休息の重心を時刻順にリストに格納しました")
	print(sys._getframe().f_code.co_name, "正常終了\n")
	return rest_list

#休息でない行動のみのリストにする
"""
Parameter
	zipped_list	: 圧縮後のリスト
return
	walk_list	: 行動のリスト（他の要素はなにも変えない）
"""
def make_walk_data(zipped_list):
	print(sys._getframe().f_code.co_name, "実行中")
	print("休息以外の行動のみを取り出します---")
	walk_list = []
	for row in zipped_list:
		if (row[4] != 0):
			walk_list.append(row)
	print("---休息以外の行動のみをリストに格納しました")
	print(sys._getframe().f_code.co_name, "正常終了\n")
	return walk_list

#歩行の変化点検出を行う
#どのように行うかは未定，歩行の前後にウォーミングアップのような時間が存在するため
"""
	t_list	:圧縮後の時刻のリスト
	p_list	:圧縮後の位置情報のリスト
	d_list	:圧縮後の距離のリスト
	l_list	:圧縮後の暫定的なラベルのリスト
"""
"""
def devide_walk_point(t_list, p_list, d_list, v_list, l_list):
"""

#特徴をCSVにして出力する (圧縮が既に行われている前提) 
"""
Parameter
	filename	:ファイルのパス
	t_list	:圧縮後の時刻のリスト
	p_list	:圧縮後の位置情報のリスト
	d_list	:圧縮後の距離のリスト
	l_list	:圧縮後の暫定的なラベルのリスト
"""
def output_feature_info(filename, t_list, p_list, d_list, v_list, l_list):
	print(sys._getframe().f_code.co_name, "実行中")
	feature_list =[]

	###登録に必要な変数###
	previous_lat = p_list[0][0]
	previous_lon = p_list[0][1]
	previous_rest_time = None

	#####登録情報#####
	lat = 0.0
	lon = 0.0
	time_length = None #現在の休息の時間
	moving_distance = None #休息間の距離
	moving_direction = None #次の休息への移動方向
	previous_rest_length = None #前の休息の長さ
	previous_rest_interval = None #前の休息からの間隔

	print("特徴を計算します---")
	#####登録#####
	for time, pos, dis, vel, label in zip(t_list, p_list, d_list, v_list, l_list):
		time_length = (time[1] - time[0]).total_seconds()
		lat = pos[0]
		lon = pos[1]

		moving_distance, moving_direction = geo.get_distance_and_direction(previous_lat, previous_lon, lat, lon, True) #前の重心との直線距離
		sum_of_distance = dis # 行動内での移動距離
		if (previous_rest_time is not None):
			previous_rest_interval = (time[0] - previous_rest_time).total_seconds() / 5 + 1

		if (label == 0): # 休息
			previous_rest_length = (time[1] - time[0]).total_seconds() / 5 + 1
			previous_rest_time = time[1] # 前の休息が終わった時間
			previous_rest_interval = 0

		###登録###
		feature_list.append([time, lat, lon, time_length, sum_of_distance, vel[0], vel[1], vel[2], moving_distance, moving_direction, previous_rest_length, previous_rest_interval, label])
		###引継###
		previous_lat = lat
		previous_lon = lon
	print("---特徴を計算しました")

	print(filename + "に出力します---")
	#####出力#####
	with open(filename, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(("Start time", "Latitude", "Longitude", "Continuous time", "Moving amount", "Average velocity", "Max velocity", "Min velocity", "Moving distance", "Moving direction", "Last rest length", "Last rest interval", "Label"))
		for feature in feature_list:
			writer.writerow(feature)
	print("---" + filename + "に出力しました")
	print(sys._getframe().f_code.co_name, "正常終了\n")
	return

#3次元のデータを主成分分析し，2次元にする
def reduce_dim_from3_to2(x, y, z):
    print("今から主成分分析を行います")
    features = np.array([x.values, y.values, z.values]).T
    pca = skd.PCA()
    pca.fit(features)
    transformed = pca.fit_transform(features)
    print("累積寄与率: ", pca.explained_variance_ratio_)
    print("主成分分析が終了しました")
    return transformed[:, 0], transformed[:, 1]

#移動速度に対して分類を行い視覚化を可能にする
"""
Parameter
	v_list	:速さのリスト
	l_list	:ラベルのリスト．Noneであれば速さだけで分類を行い，Noneでなければこのラベルを元に分類を行う
return
	それぞれが振り分けられたクラスタ ("red", "green", "blue")
"""
def calassify_velocity(v_list, graze = 0.069, walk = 0.18, l_list=None):
	data_list = []
	if (l_list is None):
		for data in v_list:
			if(data < graze):
				data_list.append("red")
			elif(data < walk):
				data_list.append("green")
			else:
				data_list.append("blue")
	else:
		for data, label in zip(v_list, l_list):
			if (label == "R"):
				data_list.append("red") # Rest
			elif (label == "G"):
				data_list.append("green") # Graze
			elif (label == "W"):
				data_list.append("blue") # Walk
			elif (label == "O"):
				data_list.append("yellow") # Other
			elif (label == "N"):
				data_list.append("magenta") # None
			else:
				data_list.append("black") #なぜかどれにも当てはまらない(デバッグ用)
	return data_list

#圧縮した休息とその他をラベル付きで解凍する
"""
Parameter
	zipped_t_list	:圧縮された時間のリスト
	zipped_l_list	:圧縮により求められたラベルのリスト
	t_list	:圧縮前の時間のリスト（いつも5sの間隔で作られている訳ではないので元の時間のリストの時間を参照する）
Return
	l_list	:解凍したラベルのリスト
"""
def decode(t_list, zipped_t_list, zipped_l_list):
	print(sys._getframe().f_code.co_name, "実行中")
	index = 0
	l_list = []

	start = zipped_t_list[0][0]
	end = zipped_t_list[0][1]
	label = zipped_l_list[0]
	for time in t_list:
		if (start <= time and time <= end):
			l_list.append(label)
		if (len(l_list) == len(t_list)):
			break
		if (end <= time):
			index += 1
			start = zipped_t_list[index][0]
			end = zipped_t_list[index][1]
			label = zipped_l_list[index]

	print(sys._getframe().f_code.co_name, "正常終了\n")
	return l_list

if __name__ == '__main__':
	filename = "features.csv"
	start = datetime.datetime(2018, 12, 30, 0, 0, 0)
	end = datetime.datetime(2018, 12, 31, 0, 0, 0)
	time_list, position_list, distance_list, velocity_list, angle_list = read_gps(20158, start, end) #2次元リスト (1日分 * 日数分)
	for (t_list, p_list, d_list, v_list, a_list) in zip(time_list, position_list, distance_list, velocity_list, angle_list):
		print(len(t_list))
		t_list, p_list, d_list, v_list, a_list = select_use_time(t_list, p_list, d_list, v_list, a_list) #日本時間に直した上で牛舎内にいる時間を除く
		#t_list, d_list, a_list = ma.convo_per_minutes(t_list, d_list, a_list, 3) #畳み込み
		plot_velocity_data(t_list, v_list)
		
		c_list = calassify_velocity(v_list) #クラスタ分けを行う (速さを3つに分類しているだけ)
		scatter_plot(t_list, v_list, c_list) #時系列で速さの散布図を表示

		zipped_list = zipping(t_list, p_list, d_list, v_list) # 圧縮する
		zipped_act_list = make_walk_data(zipped_list)
		output_feature_info(filename, [row[0] for row in zipped_act_list], [row[1] for row in zipped_act_list], [row[2] for row in zipped_act_list], [row[3] for row in zipped_act_list], [row[4] for row in zipped_act_list]) # 特徴を出力する
		
		#display = disp.Adjectory(True)
		#display.plot_moving_ad(p_list) # 移動の軌跡をプロット
		#display = disp.Adjectory(True)
		#zipped_rest_list = make_rest_data(zipped_list) # 描画用に休息時間と重心だけのリストにする
		#display.plot_rest_place(zipped_rest_list) # 休息の場所の分布のプロット
		
		#c_list = calassify_velocity([row[3] for row in zipped_list]) # クラスタ分けを行う (速さを3つに分類しているだけ)
		#scatter_plot([row[0] for row in zipped_list], [row[2] for row in zipped_list], c_list) # 時系列で速さの散布図を表示
		"""
		df = pd.read_csv(filepath_or_buffer = "features.csv", encoding = "utf-8", sep = ",", header = 0, usecols = [0,3,4,5,6,8,9,10], names=('A', 'D', 'E', 'F', 'G', 'I', 'J', 'K'))
		b, c = reduce_dim_from3_to2(df['E'], df['F'], df['I'])
		observation = np.stack([b, c]).T
		interface = hmm.hmm_interface(3)
		interface.train_data(observation)
		print("遷移行列: ",interface.transition_matrix)
		print("出力期待値: ",interface.means)
		print("初期確率: ",interface.init_matrix)
		result = interface.predict_data(observation)
		c_list = decode(t_list, [row[0] for row in zipped_list], result)
		scatter_plot(t_list, v_list, c_list) # 時系列で速さの散布図を表示
		"""