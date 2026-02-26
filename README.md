## 兴趣推理与画像融合的序列推荐系统

### Config

1. 安装依赖

   ```bash
   pip install requirement.txt
   ```

2. 下载数据集

​	 [Download Link](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)，放入`./data/raw`

3. 下载Embedding模型，`bge-large-en-v1.5`和  LLMs   `LLaMA 3 8B Instruct`

### Run

1. 处理数据

   ```python
   python 0_data_processing.py
   ```

2. 生成语义Embedding

   ```python
   python 1_generate_embedding.py
   ```

3. 分割用户兴趣，生成用户画像

   ```python
   python 2_generate_profile.py
   ```


4. 生成用户画像和`Embedding`

   ```python
   python 3_pca_embedding.py
   ```

5. 训练

   ```python
   python main.py
   ```

### Dataset

|    Name     |  User   |  Item   | Interaction |
| :---------: | :-----: | :-----: | :---------: |
| Electronics | 24,181  | 23,771  |   210,988   |
|   Sports    | 20,697  | 21,306  |   169,636   |
|    Books    | 107,432 | 155,426 |  1,539,499  |



### Results

#### Electronics

|          Model           | NDCG@10 | HR@10  | NDCG@20 | HR@20  |
| :----------------------: | :-----: | :----: | :-----: | :----: |
|         vanilla          | 0.1971  | 0.3116 | 0.2262  | 0.4271 |
|    vanilla + llm_init    | 0.2313  | 0.3640 | 0.2614  | 0.4836 |
| vanilla+llm_init+session | 0.2426  | 0.3826 | 0.2732  | 0.5044 |

#### Sports

|           Model           | NDCG@10 | HR@10  | NDCG@20 | HR@20  |
| :-----------------------: | :-----: | :----: | :-----: | :----: |
|          vanilla          | 0.2226  | 0.3118 | 0.2485  | 0.4147 |
|    vanilla + llm_init     | 0.2792  | 0.4018 | 0.3079  | 0.5162 |
| vanilla+llm_init +session | 0.2903  | 0.4179 | 0.3184  | 0.5294 |

#### Books

|           Model            | NDCG@10 | HR@10  | NDCG@20 | HR@20  |
| :------------------------: | :-----: | :----: | :-----: | :----: |
|          vanilla           | 0.3886  | 0.5426 | 0.4141  | 0.6437 |
|     vanilla + llm_init     |         |        |         |        |
| vanilla+llm_init + session |         |        |         |        |




