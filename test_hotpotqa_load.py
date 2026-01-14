import sys
import os
import numpy as np
import pickle
from unittest.mock import MagicMock

# 模拟 jamming_utils 需要的环境
sys.path.append('corpus_poisoning')
sys.path.append('corpus_poisoning/src')

import jamming_utils

# Mock embedding model dictionary
mock_tokenizer = MagicMock()
def mock_tokenize(text, **kwargs):
    # 返回一个字典，模拟分词器的输出
    count = len(text) if isinstance(text, list) else 1
    return {
        'input_ids': [[1, 2, 3]] * count,
        'attention_mask': [[1, 1, 1]] * count
    }
mock_tokenizer.side_effect = mock_tokenize

emb_dict = {
    'emb_model_name': 'gtr-base',
    'score_func': lambda x, y: np.zeros((x.shape[0], y.shape[0])),
    'score_func_name': 'cos_sim',
    'tokenizer': mock_tokenizer,
    'max_seq_length': 128,
    'pad_to_max_length': True,
    'device': 'cpu'
}

# Mock get_embedding
def mock_get_embedding(text, emb_dict, use_query_model=False):
    count = len(text) if isinstance(text, list) else 1
    return np.random.rand(count, 768)

jamming_utils.get_embedding = mock_get_embedding

# 为了防止测试时真的去下载数据，我们确信数据已经存在
# 但我们也 mock 掉下载工具以防万一
import corpus_poisoning.src.beir.beir.util as beir_util
beir_util.download_and_unzip = MagicMock(return_value="./corpus_poisoning/datasets/hotpotqa")

try:
    print("正在测试 hotpotqa 的 load_queries 函数...")
    # 只加载 2 个查询进行快速验证
    dataloader, q_embs, q_embs_names, corpus = jamming_utils.load_queries('hotpotqa', 2, emb_dict)
    
    print("\n验证成功！")
    print(f"DataLoader 样本数: {len(dataloader.dataset)}")
    print(f"查询向量矩阵形状 (q_embs): {q_embs.shape}")
    print(f"查询名称数量: {len(q_embs_names)}")
    print(f"语料库 (Corpus) 大小: {len(corpus)}")
    
    # 检查返回的数据结构是否正确
    sample = next(iter(dataloader))
    print(f"数据样例 keys: {sample[0].keys()}")
    
except Exception as e:
    print(f"\n测试失败，错误信息: {e}")
    import traceback
    traceback.print_exc()
    exit(1)