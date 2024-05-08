from __future__ import division, print_function, absolute_import
import torch
from torch.nn import functional as F
import numpy as np


def getemb(data):
    return data["inference_feat"]

def computedistence(x, y, metric="euclidean"):
    distance = None
    if metric == "euclidean":
        distance = torch.sqrt(torch.sum(torch.square(x - y)))
    else:
        xx = F.normalize(x.view(1, -1))
        yy = F.normalize(y.view(1, -1))
        distance = F.cosine_similarity(xx, yy)
    
    return distance

def compareid(data, dict, pid, threshold_value, metric="euclidean"):
    probe_name = pid.split("-")[0]
    embs = getemb(data)
    min = threshold_value
    id = None
    dic={}
    for key in dict:
        if key == probe_name:
            continue
        for subject in dict[key]:
            for type in subject:
                for view in subject[type]:
                    value = subject[type][view]
                    distance = computedistence(embs["embeddings"], value, metric=metric)
                    gid = key + "-" + str(type)
                    gid_distance = (gid, distance)
                    dic[gid] = distance
                    if distance.float() < min:
                        id = gid
                        min = distance.float()
    dic_sort= sorted(dic.items(), key=lambda d:d[1], reverse = False)
    if id is None:
        print("############## no id #####################")
    return id, dic_sort


def comparefeat(embs, gallery_feat: dict, pid, threshold_value, metric="euclidean"):
    """Compares the distance between features

    Args:
        embs (Tensor): Embeddings of person with pid
        gallery_feat (dict): Dictionary of features from gallery
        pid (str): The id of person in probe
        threshold_value (int): Threshold
    Returns:
        id (str): The id in gallery
        dic_sort (dict): Recognition result sorting dictionary
    """
    print(f"probe_pid: {pid}")
    probe_name = pid.split("-")[0]
    min = threshold_value
    id = None
    dic={}
    for key in gallery_feat:
        if key == probe_name:
            continue
        for subject in gallery_feat[key]:
            for type in subject:
                for view in subject[type]:
                    value = subject[type][view]
                    distance = computedistence(embs, value, metric=metric)
                    gid = key + "-" + str(type)
                    gid_distance = (gid, distance)
                    dic[gid] = distance
                    print(f"key: {key}, g_pid: {type}, distance: {(distance.float(), 3)}")
                    if distance.float() < min:
                        id = gid
                        min = distance.float()
    dic_sort= sorted(dic.items(), key=lambda d:d[1], reverse = False)
    if id is None:
        print("############## no id #####################")
    return id, dic_sort
