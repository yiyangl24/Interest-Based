import copy

import torch
import pickle
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel, AutoTokenizer, AutoConfig


class MyDataset(Dataset):

    def __init__(self, item_prompt):
        self.items = list(item_prompt.items())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


if __name__ == '__main__':

    dataname_list = ['Electronics', 'Sports_and_Outdoors', 'Books']

    for dataname in dataname_list:

        processd_dir = f'./data/processed/{dataname}/'

        device = torch.device('cuda:1')

        llm_path = '../bge-large-en-v1.5'

        llm = AutoModel.from_pretrained(llm_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(llm_path)

        config = AutoConfig.from_pretrained(llm_path)


        llm.eval()

        with open(processd_dir + dataname + '_meta.pkl', 'rb') as f:
            item_prompt = pickle.load(f)

        dataset = MyDataset(item_prompt)

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: list(zip(*batch))
        )

        item_embedding = {}

        instruct_template = "Represent this item for recommendation:\n"

        max_input_length = 4096

        for asins, prompts in tqdm(dataloader):
            prompts = list(prompts)
            prompts = [instruct_template + p for p in prompts]
            inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=max_input_length).to(device)
            with torch.no_grad():
                outputs = llm(**inputs)
                embeddings = outputs[0][:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            for asin, i_embedding in zip(asins, embeddings):
                item_embedding[asin] = i_embedding.detach().cpu().tolist()

        with open(processd_dir + dataname + '.pkl', 'rb') as f:
            _, _, _, _, item_dict, item_cnt = pickle.load(f)

        item_dict = item_dict['id2str']

        embedding_dim = len(list(item_embedding.values())[0])

        embedding_list = []
        for iid in range(1, item_cnt + 1):
            asin = item_dict[iid]
            if asin in item_embedding.keys():
                embedding_list.append(item_embedding[asin])
            else:
                embedding_list.append([0] * embedding_dim)

        embedding_list = np.array(embedding_list)

        with open(processd_dir + 'item_embedding.pkl', 'wb') as f:
            pickle.dump(embedding_list, f, pickle.HIGHEST_PROTOCOL)

        torch.cuda.empty_cache()



