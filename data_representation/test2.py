import pickle
import numpy as np

"""
file_path = "../dataset/represented_data/events/events_test/nb8/sample.pkl"  # 読みたい pkl ファイルのパス

with open(file_path, "rb") as f:  # "rb" = バイナリ読み込みモード
    data = pickle.load(f)

print(data[:10])
"""

file_path = "../dataset/represented_data/tuneidx/tuneidx_test/nb8/sample.npz"  # 読みたい npz ファイルのパス

data = np.load(file_path)
print(data['arr_0'][:10])  # 'arr_0' は npz ファイル内の配列の名前