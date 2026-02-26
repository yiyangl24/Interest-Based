import os
import copy
import json
import gzip
import random
import pickle
import requests
import numpy as np
import matplotlib.pyplot as plt

from uitls import *
from tqdm import tqdm
from collections import defaultdict


def amazon_dataset(dataname, rating_score):
    '''
    reviewerID: ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin: ID of the product, e.g. 0000013714
    reviewerName: name of the reviewer
    helpful: helpfulness rating of the review, e.g. 2 or 3
    reviewText: text of the review, e.g. "I bought this for my husband who plays the piano. ..."
    overall: rating of the product, e.g. 5.0
    summary: summary of the review, e.g. "Heavenly Highway Hymns"
    unixReviewTime: time of the review (unix time), e.g. 1252800000,
    reviewTime: time of the review (raw), e.g. "09 13, 2009"
    '''
    records = []
    data_file = './data/raw/' + str(dataname) + '.json.gz'
    g = gzip.open(data_file, 'rb')
    for l in g:
        inter = json.loads(l.decode())
        if float(inter['overall']) <= rating_score:
            continue
        user = inter['reviewerID']
        item = inter['asin']
        time = inter['unixReviewTime']
        records.append([user, item, int(time)])
    return records


def filter_time(data, t_min, t_max):
    return [inter for inter in data if t_min < inter[2] < t_max]


def check_Kcore(data, user_core, item_core):
    user_cnt = defaultdict(int)
    item_cnt = defaultdict(int)

    for uid, iid, t in data:
        user_cnt[uid] += 1
        item_cnt[iid] += 1

    isKcore = min(user_cnt.values()) >= user_core and min(item_cnt.values()) >= item_core

    return user_cnt, item_cnt, isKcore


def filter_Kcore(data, user_core, item_core):

    user_cnt, item_cnt, isKcore = check_Kcore(data, user_core, item_core)

    while not isKcore:

        cur_data = []

        for uid, iid, t in tqdm(data):
            if user_cnt[uid] >= user_core and item_cnt[iid] >= item_core:
                cur_data.append([uid, iid, t])

        user_cnt, item_cnt, isKcore = check_Kcore(cur_data, user_core, item_core)

        data = cur_data

    user_set = set(user_cnt.keys())
    item_set = set(item_cnt.keys())

    return user_set, item_set, data


def build_dict(user_set, item_set, data):
    user_dict = {
        'str2id': dict(zip(list(user_set), range(1, len(user_set) + 1))),
        'id2str': dict(zip(range(1, len(user_set) + 1), list(user_set))),
    }
    item_dict = {
        'str2id': dict(zip(list(item_set), range(1, len(item_set) + 1))),
        'id2str': dict(zip(range(1, len(item_set) + 1), list(item_set))),
    }

    tmp_data = defaultdict(list)
    for uid, iid, t in data:
        uid = user_dict['str2id'][uid]
        iid = item_dict['str2id'][iid]
        tmp_data[uid].append([iid, t])

    final_data = defaultdict(list)
    for uid, inter in tmp_data.items():
        inter.sort(key=lambda x: x[1])
        final_data[uid] = [iid for iid, _ in inter]

    item_cnt = len(item_set)

    return final_data, user_dict, item_dict, item_cnt


def amazon_meta(dataname, item_dict):
    description = {}
    data_file = './data/raw/meta_' + str(dataname) + '.json.gz'
    g = gzip.open(data_file, 'rb')
    asin_set = set(item_dict['str2id'].keys())
    for l in g:
        desc = eval(l)
        if desc['asin'] in asin_set:
            description[desc['asin']] = desc
    return description


def get_attribute(i_prompt, i_attr, i_desc):
    r = "unknown"
    if i_attr in i_desc.keys() and len(i_desc[i_attr]) <= 100:
        r = str(i_desc[i_attr])
    i_prompt = i_prompt.replace(f"<{i_attr.upper()}>", r)
    return i_prompt


def get_feature(i_prompt, i_feature, i_desc):
    if i_feature not in i_desc.keys():
        return ""
    assert isinstance(i_desc[i_feature], list)
    r = "; ".join(i_desc[i_feature])
    i_prompt = i_prompt.replace(f"<{i_feature.upper()}>", r)
    return i_prompt[:128]


if __name__ == '__main__':

    # following LLM-SRec

    fix_random_seed(seed=42)

    sample_rate = {
        'Electronics': 0.05,
        'Beauty': 1,
        'Books': 0.075,
        'Sports_and_Outdoors': 0.1,
    }

    for dataname in ['Sports_and_Outdoors', 'Books']:

        processd_dir = f'./data/processed/{dataname}/'

        if not os.path.exists(processd_dir + dataname + '.pkl'):

            data = amazon_dataset(dataname, 0)

            #### 采样
            all_users = list(set([record[0] for record in data]))
            sampled_cnt = int(len(all_users) * sample_rate[dataname])
            sampled_users = set(np.random.choice(all_users, sampled_cnt, replace=False))
            data = [record for record in data if record[0] in sampled_users]

            # 循环过滤, 用户的交互次数不少于 5 次, 物品不少于 3 次
            user_set, item_set, data = filter_Kcore(data, 5, 3)

            print(f"{dataname} user: {len(user_set)} item: {len(item_set)} inter: {len(data)}")

            # 按时间排序, 构建索引 dict
            data, user_dict, item_dict, item_cnt = build_dict(user_set, item_set, data)

            meta_data = amazon_meta(dataname, item_dict)

            title_list = []

            for iid in range(1, item_cnt + 1):
                asin = item_dict['id2str'][iid]
                if asin in meta_data.keys():
                    title_list.append(meta_data[asin]['title'][:100])
                else:
                    title_list.append('no name')

            assert len(title_list) == item_cnt

            with open(processd_dir + dataname + '.pkl', 'wb') as f:
                pickle.dump((data, meta_data, title_list, user_dict, item_dict, item_cnt), f, pickle.HIGHEST_PROTOCOL)


        with open(processd_dir + dataname + '.pkl', 'rb') as f:
            data, meta_data, title_list, user_dict, item_dict, item_cnt = pickle.load(f)

        # make_prompt
        prompt_template = f"The {dataname.lower()} item has following attributes: \n name is <TITLE>; brand is <BRAND>; price is <PRICE>, rating is <DATE>, price is <PRICE>. \n"
        feature_template = "The item has following features: <CATEGORY>."
        description_template = "\nThe item has following descriptions: <DESCRIPTION>."

        item_prompt = {}

        for asin, desc in meta_data.items():

            i_prompt = copy.deepcopy(prompt_template)
            i_prompt = get_attribute(i_prompt, 'title', desc)
            i_prompt = get_attribute(i_prompt, 'brand', desc)
            i_prompt = get_attribute(i_prompt, 'date', desc)
            i_prompt = get_attribute(i_prompt, 'price', desc)

            f_prompt = copy.deepcopy(feature_template)
            f_prompt = get_feature(f_prompt, 'category', desc)

            d_prompt = copy.deepcopy(description_template)
            d_prompt = get_feature(d_prompt, 'description', desc)

            item_prompt[asin] = i_prompt + f_prompt + d_prompt

        with open(processd_dir + dataname + '_meta.pkl', 'wb') as f:
            pickle.dump(item_prompt, f, pickle.HIGHEST_PROTOCOL)



