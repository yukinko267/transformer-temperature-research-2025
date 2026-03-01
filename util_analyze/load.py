# Analyze_util.py
import pickle

def load_attn_pkls(pkl_paths):
    """
    pkl_paths: list[str]  読み込みたい sample.pkl のパス列
    return: list[dict]   各 pkl の中身
    """
    data_list = []
    for path in pkl_paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
        data_list.append(data)
    return data_list
