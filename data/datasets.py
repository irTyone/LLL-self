import torch
from utils.utils import clean_english_text,clean_chinese_text,remove_html_and_newlines
from torch.utils.data import Dataset, DataLoader
import os
import json

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128, stride=1, file_idx=0,lang="ch"):
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
           
            text=data_file.get("text")   # 添加 CLS 和 SEP token
            contents.append(f"{self.tokenizer.cls_token} {text} {self.tokenizer.sep_token}")

        all_text = " ".join(contents)
        tokens = self.tokenizer.encode(all_text)
        # print(tokens)
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

class LazyTextDataset(Dataset):
    """
    完全 lazy 版本 Dataset：
    - 文件按 file_idx 控制跳过已训练文件
    - 不在内存中保存整个 token 序列
    - 在 __getitem__ 时按需读取文本并 tokenize
    """
    def __init__(self, folder_or_file, tokenizer, block_size=256, stride=10, lang='ch', file_idx=0):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride
        self.lang = lang

        # 收集文件
        if os.path.isfile(folder_or_file):
            self.files = [folder_or_file]
        else:
            self.files = sorted([os.path.join(folder_or_file, f) for f in os.listdir(folder_or_file)])
        # 跳过已训练文件
        self.files = self.files[file_idx:]

        # 构建索引：每个元素是 (file_path, text_idx, start_pos)
        self.block_index = []
        for fpath in self.files:
            with open(fpath, 'r', encoding='utf-8') as f:
                data_file = json.load(f)
            for text_idx, data in enumerate(data_file):
                print(data,type(data))
                if self.lang == 'ch':
                    text = clean_chinese_text(data.get("text", ""))
                else:
                    text = clean_english_text(data.get("text", ""))
                # 添加 CLS / SEP
                text = f"{self.tokenizer.cls_token} {text} {self.tokenizer.sep_token}"
                token_ids = self.tokenizer.encode(text)
                vocab_size = self.tokenizer.vocab_size
                token_ids = [min(max(t, 0), vocab_size - 1) for t in token_ids]

                # 按 block_size + stride 构建索引
                for start in range(0, len(token_ids) - self.block_size, self.stride):
                    self.block_index.append((fpath, text_idx, start, len(token_ids)))

    def __len__(self):
        return len(self.block_index)

    def __getitem__(self, idx):
        fpath, text_idx, start, text_len = self.block_index[idx]

        # 读取文本并 tokenize
        with open(fpath, 'r', encoding='utf-8') as f:
            data_file = json.load(f)
            data = data_file[text_idx]

        if self.lang == 'ch':
            text = clean_chinese_text(data.get("text", ""))
        else:
            text = clean_english_text(data.get("text", ""))

        text = f"{self.tokenizer.cls_token} {text} {self.tokenizer.sep_token}"
        token_ids = self.tokenizer.encode(text)
        vocab_size = self.tokenizer.vocab_size
        token_ids = [min(max(t, 0), vocab_size - 1) for t in token_ids]

        end = min(start + self.block_size, len(token_ids))
        input_ids = token_ids[start:end]
        target_ids = token_ids[start + 1:end + 1]

        # padding
        if len(input_ids) < self.block_size:
            pad_len = self.block_size - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            target_ids += [self.tokenizer.pad_token_id] * pad_len

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        return input_ids, target_ids