#-*- encoding:utf-8 -*-
import numpy as np
import os,sys
import datetime
import pandas as pd
import cv2.cv2 as cv2
import copy
import pdb

# 自作クラス
# import visualization.place_plot as place_plot

class PlacePlotter:
    """ 1枚の画像を作成するクラス, imageはnp.ndarray, save_imageで画像を保存 """
    image:np.ndarray
    __size = (633, 557) # (width, height)
    __top_left = (34.88328, 134.86187) # (latitude, longitude)
    __bottom_right = (34.88207, 134.86357) # (latitude, longitude)

    def __init__(self, back:bool, image=None):
        """ Parameter
                back	: bool  背景に衛星画像を表示するかどうか """
        if(back):
            if (image is None):
                self.image = cv2.imread("./visualization/image/background.jpg")
            else:
                self.image = image
        else:
            self.image = cv2.imread('RGBA',self.__size,(255,255,255,255))
    
    def plot_places(self, pos_list, caption_list=None, color_list=None, label="", former_list=[], is_lined=False):
        """ リスト型の複数の位置情報を描画する
            Parameter
                pos_list        : list  (lat, lon) の2要素の2次元
                caption_list    : list  画像に表示するラベル (牛の個体番号など)
                color_list      : list  塗りつぶしの色のリスト
                label           : str   右下に表示するもの（時間情報など）
                former_list     : list  前時刻のpos_listなど. 軌跡を知りたい場合などに使用する．指定がなければ使用しない
                is_lined        : bool  同じ色同士の点を線で結ぶかどうか """
        colors = [(127, 0, 0), (0, 0, 127), (127, 0, 127), (127, 127, 127), (127, 127, 0), (0, 127, 127), (0, 127, 0), \
            (255, 0, 0), (0, 0, 255), (255, 0, 255), (255, 255, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0)] # 緑と同系色はなるべく避ける
        for i, pos in enumerate(pos_list):
            color = colors[color_list[i]%len(colors)] if color_list is not None else (255,255,255) # 色の指定があれば色を付けられる
            before_pos = former_list[i] if len(former_list) != 0 else pos
            x_f, y_f = self._draw_circle(2, float(before_pos[0]), float(before_pos[1]), color) # 前回の場所
            x, y = self._draw_circle(2, float(pos[0]), float(pos[1]), color) # 今回の場所
            cv2.line(self.image, (x_f, y_f), (x, y), color) # 線を引く
            caption = str(caption_list[i]) if caption_list is not None else "" # キャプションの指定があればキャプションを付けられる
            cv2.putText(self.image, caption, (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        if (is_lined): # ノード同士を線で結ぶ場合
            for i, pos1 in enumerate(pos_list):
                for j, pos2 in enumerate(pos_list):
                    if (i < j):
                        if (color_list[i] == color_list[j]):
                            color = colors[color_list[i]%len(colors)] if color_list is not None else (255,255,255)
                            x1, y1 = self._draw_circle(2, float(pos1[0]), float(pos1[1]), color)
                            x2, y2 = self._draw_circle(2, float(pos2[0]), float(pos2[1]), color)
                            cv2.line(self.image, (x1, y1), (x2, y2), color) # 2地点間を線で結ぶ
        cv2.putText(self.image, label, (5, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
        return

    def _draw_circle(self, radius, latitude, longitude, color):
        """ 円を描く
            Parameter
                draw	:ImageDraw.Draw :図形描画用オブジェクト
                radius  : 円の半径
                latitude, longitude :緯度・経度
                color   : 色
            Return
                x, y    : プロットの場所（ピクセル） """
        width = self.__bottom_right[1] - self.__top_left[1] # 正
        height = self.__bottom_right[0] - self.__top_left[0] # 負
        x = int(((longitude - self.__top_left[1]) / width) * self.__size[0])
        y = int(((latitude - self.__top_left[0]) / height) * self.__size[1])
        if((0 <= x and x <= self.__size[0]) and (0 <= y and y <= self.__size[1])):
            cv2.circle(self.image, (x, y), radius, color, thickness=-1)
        return x, y
    
    def get_image(self):
        return self.image

    def save_image(self, filename):
        cv2.imwrite(filename, self.image)
        return


class PlotMaker:
    """ 複数枚の画像から位置プロット動画または軌跡画像を作成するクラス """
    video_filename: str # output用
    image_filename: str # output用
    width: int
    height: int
    caption_list:list # ノードにつけるキャプションのリスト
    color_list: list # ノードの色のリスト
    is_lined: bool # ノード同士を線で結ぶかどうか

    def __init__(self, caption_list=None, color_list=None, video_filename="", image_filename="", is_lined=False):
        self.video_filename = "./visualization/" + video_filename
        self.image_filename = "./visualization/" + image_filename
        size = cv2.imread("./visualization/image/background.jpg").shape
        self.height = size[0]
        self.width = size[1]
        self.caption_list = caption_list
        self.color_list = color_list
        self.is_lined = is_lined

    def make_movie(self, df:pd.DataFrame, frame_rate=10.0, disp_adj=False):
        """ 動画を作成する
            df  : position_information.synchronizer.Synchoronizer.cows_dataの形式
            disp_adj    : bool 軌跡を更新表示するかどうかのフラグ """
        # 画像の系列の取得
        images = self._make_sequence_images(df, "video", disp_adj)
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        video  = cv2.VideoWriter(self.video_filename, fourcc, frame_rate, (self.width, self.height))
        # 画像を繋げて動画にする
        for img in images:
            video.write(img)
        # 出力
        video.release()
        print("動画を出力しました -> ", self.video_filename)
        return

    def make_adjectory(self, df:pd.DataFrame):
        """ 軌跡画像を作成する
            df  : position_information.synchronizer.Synchoronizer.cows_dataの形式 """
        images = self._make_sequence_images(df, "image", True) # 画像系列の取得
        adjectory_image = images[len(images)-1] # 最後の画像が軌跡画像
        cv2.imwrite(self.image_filename, adjectory_image) # 軌跡画像を保存
        print("軌跡画像を出力しました -> ", self.image_filename)
        return

    def _make_sequence_images(self, df:pd.DataFrame, filetype:str, disp_adj:bool):
        """ 指定時刻の画像データを系列として保持する
            filetype    : video or image
            disp_adj    : 背景画像を軌跡にするか否か """
        images = []
        start = df.index[0]
        end = df.index[len(df.index)-1]
        if (filetype == "video"):
            self.video_filename += start.strftime("%Y%m%d/")
            self._confirm_dir(self.video_filename) # ディレクトリを作成
            self.video_filename += start.strftime("%H%M%S-") + end.strftime("%H%M%S.mp4") # ファイル名を決定
        if (filetype == "image"):
            self.image_filename += start.strftime("%Y%m%d/")
            self._confirm_dir(self.image_filename) # ディレクトリを作成
            self.image_filename += start.strftime("%H%M%S-") + end.strftime("%H%M%S.jpg") # ファイル名を決定
        former_data, former_image = None, None
        for time, data in df.iterrows():
            image = self._make_image(data, time, former_df=former_data, image=former_image) if disp_adj \
                else self._make_image(data, time, former_df=None, image=None) # 背景を指定するか否かdisp_adjに従って指定し画像を作成
            images.append(image)
            former_data, former_image = data, copy.deepcopy(image) # deepcopyしないと画像が参照渡しになってしまうので注意
        return images

    def _make_image(self, df:pd.DataFrame, time:datetime.datetime, former_df=None, image=None):
        """ 1時刻から画像を作成する """
        plotter = PlacePlotter(True, image=image)
        former_pos_list = []
        if (former_df is not None):
            for _, data in former_df.iteritems():
                former_pos = (data[0], data[1])
                former_pos_list.append(former_pos)
        pos_list = []
        for _, data in df.iteritems():
            pos = (data[0], data[1])
            pos_list.append(pos)
        plotter.plot_places(pos_list, caption_list=self.caption_list, color_list=self.color_list, label=time.strftime("%Y/%m/%d %H:%M:%S"), former_list=former_pos_list, is_lined=self.is_lined)
        output_image = plotter.get_image()
        return output_image

    def _confirm_dir(self, dir_path):
        """ ファイルを保管するディレクトリが既にあるかを確認し，なければ作成する """
        if (os.path.isdir(dir_path)):
            return
        else:
            os.makedirs(dir_path)
            print("ディレクトリを作成しました", dir_path)
            return

if __name__ == "__main__":
    os.chdir('../') # カレントディレクトリを一階層上へ
    print(os.getcwd())
    sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
    import synchronization.functions.utility as my_utility
    import synchronization.community_creater as community_creater
    
    delta_c = 2 # コミュニティの抽出間隔 [minutes]
    delta_s = 5 # データのスライス間隔 [seconds] 
    epsilon = 12 # コミュニティ決定のパラメータ
    dzeta = 12 # コミュニティ決定のパラメータ
    leng = 1 # コミュニティ決定のパラメータ
    cows_record_file = os.path.abspath('../') + "/CowTagOutput/csv/" # 分析用のファイル
    date = datetime.datetime(2018, 6, 17, 0, 0, 0)
    start_t = datetime.datetime(2018, 6, 18, 4, 0, 0)
    end_t = datetime.datetime(2018, 6, 18, 6, 50, 0)
    target_cow_id = '20122'

    cow_id_list = my_utility.get_existing_cow_list(date, cows_record_file)
    com_creater = community_creater.CommunityCreater(date, cow_id_list)

    # interaction_graph = com_creater.make_interaction_graph(start_t, end_t, method="position", delta=delta_s, epsilon=epsilon, dzeta=dzeta)
    # community_list = com_creater.create_community(start_t, end_t, interaction_graph, delta=delta_s, leng=leng)
    # community = [com for com in community_list if target_cow_id in com][0]
    com_creater.visualize_adjectory(start_t, end_t, ['20122', '20192'], target_cow_id=target_cow_id, delta=delta_s) # 軌跡をプロット
    # com_creater.visualize_position(start_t, end_t, community, target_cow_id=target_cow_id, delta=delta_s) # 位置情報とコミュニティをプロット
