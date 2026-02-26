import copy
import os.path

import torch
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import KMeans

from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel


import time


if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    dataname_list = ['Electronics', 'Sports_and_Outdoors', 'Books']

    # params
    dataname = 'Electronics'
    processd_dir = f'./data/processed/{dataname}/'
    n_clusters = 10
    device = torch.device(f'cuda:0')
    llm_path = '../Llama-3-8B-Instruct'
    max_input_length = 1024
    batch_size = 30

    # clusering

    with open(processd_dir + 'item_embedding.pkl', 'rb') as f:
        llm_embedding = pickle.load(f)

    model = KMeans(n_clusters=n_clusters)
    model.fit(llm_embedding)
    y_pred = model.predict(llm_embedding)

    cluster_map = dict(zip(range(1, len(y_pred) + 1), y_pred))

    with open(processd_dir + 'item_cluster_map.pkl', 'wb') as f:
        pickle.dump(cluster_map, f, pickle.HIGHEST_PROTOCOL)

    with open(processd_dir + dataname + '.pkl', 'rb') as f:
        data, meta_data, title_list, user_dict, item_dict, item_cnt = pickle.load(f)

    assert max(cluster_map.keys()) == item_cnt

    # split into session
    data_session = {}
    for uid, inter in data.items():
        user_session = defaultdict(list)
        for iid in inter[:-1]:  # data leakage
            user_session[cluster_map[iid]].append(iid)
        data_session[uid] = user_session

    # generate profile
    llm = AutoModelForCausalLM.from_pretrained(llm_path, device_map=device, torch_dtype=torch.float16, load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, padding_side='left')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for name, param in llm.named_parameters():
        param.requires_grad = False

    prompt_template = {
        'local': "Assume you are a consumer who is shopping online. You have shown interests in following commdities:\n {}. The commodities are segmented by '\n'. Please conclude it not beyond 50 words. Do not only evaluate one specfic commodity but illustrate the interests overall.",
        'global': "Assume you are an consumer and there are preference demonstrations from several aspects are as follows:\n {}. Please illustrate your preference with less than 100 words."
    }

    all_session_prompt = []
    all_session_uid = []
    all_session_sid = []

    llm.eval()


    for uid, user_session in tqdm(data_session.items()):
        session_profile = []
        for sid, session in user_session.items():
            if len(session) > 0:
                r = " \n".join(title_list[iid - 1] for iid in session)
                prompt = prompt_template['local']
                prompt = prompt.format(r)
                all_session_prompt.append(prompt)
                all_session_uid.append(uid)
                all_session_sid.append(sid)

    session_file = processd_dir + 'session.pkl'

    user_session_summary = defaultdict(lambda: defaultdict(str))

    prompt, uid, sid = [], [], []

    all_session_summay = []

    for i in tqdm(range(len(all_session_prompt))):

        prompt.append(all_session_prompt[i])
        uid.append(all_session_uid[i])
        sid.append(all_session_sid[i])

        if len(uid) == batch_size or i == len(all_session_prompt) - 1:
            inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length).to(device)
            with torch.no_grad():
                outputs = llm.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.8,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
            for j in range(len(uid)):
                input_len = inputs['attention_mask'][j].sum().item()
                summary = outputs[j, input_len:]
                response = tokenizer.decode(summary, skip_special_tokens=True).strip()
                all_session_summay.append(response)
                user_session_summary[uid[j]][sid[j]] = response
            prompt, uid, sid = [], [], []

            del inputs, outputs
            torch.cuda.empty_cache()

    with open(session_file, 'wb') as f:
        pickle.dump((all_session_summay, all_session_sid, all_session_uid), f, pickle.HIGHEST_PROTOCOL)

    del all_session_prompt, all_session_uid, all_session_sid, all_session_summay

    all_user_prompt = []
    all_user_uid = []
    for uid, session_dict in user_session_summary.items():
        r = " \n".join(session_dict.values())
        prompt = prompt_template['global']
        prompt = prompt.format(r)
        all_user_prompt.append(prompt)
        all_user_uid.append(uid)

    del user_session_summary

    profile_file = processd_dir + 'profile.pkl'

    user_profile = {}

    prompt, uid = [], []

    for i in tqdm(range(0, len(all_user_prompt))):

        prompt.append(all_user_prompt[i])
        uid.append(all_user_uid[i])

        if len(uid) == batch_size or i == len(all_user_prompt) - 1:
            inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length).to(device)
            with torch.no_grad():
                outputs = llm.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.8,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
            for j in range(len(uid)):
                input_len = inputs['attention_mask'][j].sum().item()
                summary = outputs[j, input_len:]
                response = tokenizer.decode(summary, skip_special_tokens=True).strip()
                user_profile[uid[j]] = response

            prompt, uid = [], []

            del inputs, outputs
            torch.cuda.empty_cache()

    with open(profile_file, 'wb') as f:
        pickle.dump(user_profile, f, pickle.HIGHEST_PROTOCOL)

















