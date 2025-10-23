import torch
from utils.utils import clean_english_text,clean_chinese_text,remove_html_and_newlines
from torch.utils.data import Dataset, DataLoader
import os
import json

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128, stride=1, file_idx=0):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride
        self.tokens = self.read_and_tokenize(file_path, file_idx)
        

    def read_and_tokenize(self, file_path, lang='ch'):
        contents = []

        if os.path.isfile(file_path):
            files = [file_path]
        else:
            files = [os.path.join(file_path, f) for f in os.listdir(file_path)]

        for fpath in files:
            print("数据加载")
            with open(fpath, 'r', encoding='utf-8') as f:
                data_file = json.load(f)
            for data in data_file:
                if lang == 'ch':
                    text = clean_chinese_text(data.get("text", ""))
                else:
                    text = clean_english_text(data.get("text", ""))
                # 添加 CLS 和 SEP token
                contents.append(f"{self.tokenizer.cls_token} {text} {self.tokenizer.sep_token}")

        all_text = " ".join(contents)
        tokens = self.tokenizer.encode(all_text)
        
        # 保证所有 token 都在合法范围
        vocab_size = self.tokenizer.vocab_size
        tokens = [min(max(t, 0), vocab_size - 1) for t in tokens]

        return tokens

    def __len__(self):
        return max(0, (len(self.tokens) - self.block_size) // self.stride + 1)

    def __getitem__(self, idx):
       
        start = idx * self.stride
        end = start + self.block_size
        input_ids = self.tokens[start:end]
        target_ids = self.tokens[start + 1:end + 1]  # 下一步预测

        # 如果最后一块不足 block_size，补齐或者截断
        if len(input_ids) < self.block_size:
            padding = [self.tokenizer.pad_token_id] * (self.block_size - len(input_ids))
            input_ids += padding
            target_ids += padding

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        return input_ids, target_ids