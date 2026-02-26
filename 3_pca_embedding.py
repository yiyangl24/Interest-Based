import os
import torch
import pickle
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

if __name__ == '__main__':

    dataname_list = ['Electronics', 'Sports_and_Outdoors', 'Books']

    dim = 128
    batch_size = 30
    max_input_length = 512
    dataname = 'Electronics'
    device = torch.device('cuda:0')
    llm_path = '../bge-large-en-v1.5'
    processd_dir = f'./data/processed/{dataname}/'

    # 1. PCA item embedding

    with open(processd_dir + 'item_embedding.pkl', 'rb') as f:
        llm_embedding = pickle.load(f)

    pca = PCA(n_components=dim)
    pca_embedding = pca.fit_transform(llm_embedding)

    with open(processd_dir + 'item_pca_embedding.pkl', 'wb') as f:
        pickle.dump(pca_embedding, f, pickle.HIGHEST_PROTOCOL)


    # LLM
    llm = AutoModel.from_pretrained(llm_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(llm_path)

    llm.eval()

    # 2. generate session embedding
    instruct_template = "Represent the interest of this user for recommendation:\n{}"

    with open(processd_dir + 'session.pkl', 'rb') as f:
        all_session_summay, all_session_sid, all_session_uid = pickle.load(f)

    all_session_embedding = []

    prompt = []

    for i in tqdm(range(len(all_session_summay))):
        prompt.append(instruct_template.format(all_session_summay[i]))
        if len(prompt) == batch_size or i == len(all_session_summay) - 1:
            inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = llm(**inputs)
                embeddings = outputs[0][:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            for s_embedding in embeddings:
                all_session_embedding.append(s_embedding.detach().cpu().tolist())

            prompt = []

    with open(processd_dir + 'session_embedding.pkl', 'wb') as f:
        pickle.dump((all_session_uid, all_session_sid, all_session_embedding), f, pickle.HIGHEST_PROTOCOL)

    del all_session_uid, all_session_sid, all_session_summay, all_session_embedding

    # 3. generate profile embedding
    instruct_template = "Represent the profile of this user for recommendation:\n{}"

    with open(processd_dir + 'profile.pkl', 'rb') as f:
        user_profile = pickle.load(f)

    user_profile_embedding = {}

    prompt, uid = [], []

    for i, (user_id, profile) in tqdm(enumerate(user_profile.items())):
        prompt.append(instruct_template.format(profile))
        uid.append(user_id)
        if len(uid) == batch_size or i == len(user_profile) - 1:
            inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = llm(**inputs)
                embeddings = outputs[0][:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            for u, u_embedding in zip(uid, embeddings):
                user_profile_embedding[u] = u_embedding.detach().cpu().tolist()

            uid, prompt = [], []


    embedding_list = []

    user_cnt = max(user_profile_embedding.keys())

    for uid in range(1, user_cnt + 1):
        embedding_list.append(user_profile_embedding[uid])

    with open(processd_dir + 'profile_embedding.pkl', 'wb') as f:
        pickle.dump(embedding_list, f, pickle.HIGHEST_PROTOCOL)





