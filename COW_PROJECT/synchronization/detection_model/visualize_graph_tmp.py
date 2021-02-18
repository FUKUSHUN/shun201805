import os, sys
import time
import datetime
import math
import numpy as np
import pandas as pd
import pdb
import pickle
import networkx as nx # ネットワークグラフ
import matplotlib.pyplot as plt # ネットワークグラフ描画

def visualize_graph(graph, label_list, color_list:list, filename:str):
    """ インタラクショングラフを描画する
        graph:  np.array
        label_list: list    ラベルに付与するリスト
        color_list: list    ノード数分の長さのリスト．node_colorsのインデックスの数字が入る
        filename: str       セーブするファイルの名前 """
    g = _create_nxgraph(label_list, graph)
    num_nodes = len(g.nodes)
    color_list = [["lightsalmon", "lightblue"][i] for i in color_list]
    plt.figure(figsize=(9.67, 9.67))
    # 円状にノードを配置
    pos = {
        n: (np.cos(2*i*np.pi/num_nodes), np.sin(2*i*np.pi/num_nodes))
        for i, n in enumerate(g.nodes)
    }
    # ノードとエッジの描画
    normalized = np.sum(graph) / 2
    nx.draw_networkx_nodes(g, pos, node_color=color_list) # alpha: 透明度の指定
    nx.draw_networkx_labels(g, pos, font_size=15) #ノード名を付加
    # edge_labels = {(i, j): w['weight'] for i, j, w in g.edges(data=True)} # エッジの重みの描画設定
    edge_width = [10 * w['weight'] / normalized for (i, j, w) in g.edges(data=True)] # エッジの重みに従って太さを変更
    nx.draw_networkx_edges(g, pos, edge_color='black', width=edge_width)
    # nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
    plt.axis('off') #X軸Y軸を表示しない設定
    plt.savefig(filename)
    plt.close()
    return

def _create_nxgraph(label_list, matrix):
    """ 隣接行列から無向グラフを作成する
        label_list: list    ノードに付与するラベル
        matrix: np.array     隣接行列のグラフ（2次元行列で表現） """
    graph = nx.Graph()
    edges = []
    # ノードメンバを登録
    for i, label1 in enumerate(label_list):
        graph.add_node(label1)
        for j, label2 in enumerate(label_list):
            if (matrix[i,j] > 0):
                edges.append((label1, label2, math.floor(matrix[i,j] * 100) / 100))
    # エッジを追加
    graph.add_weighted_edges_from(edges)
    return graph

if __name__ == "__main__":
    dir_name = "./test/"
    community = [20192, 20197, 20215]
    matrix = np.array([
        [0, 68, 74], 
        [68, 0, 24],
        [74, 24, 0]
    ])
    visualize_graph(matrix, community, [0, 1, 1], dir_name + '20180614.jpg')

    community = [20122, 20126, 20192, 20215]
    matrix = np.array([
        [0, 107, 139, 33], 
        [107, 0, 61, 17], 
        [139, 61, 0, 37],
        [33, 17, 37, 0]
    ])
    visualize_graph(matrix, community, [1, 1, 0, 1], dir_name + '20180617.jpg')

