import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from opencc import OpenCC
from transformers import AutoTokenizer
from core.config import TOKENIER

def generate_mask(tensor :torch.Tensor):
    mask=torch.tril(torch.ones(tensor.shape[-2],tensor.shape[-1],device=tensor.device)).bool()
    return tensor.masked_fill(~mask,float('-inf'))



def remove_html_and_newlines(text):
    """去除回车符、换行符和 HTML 标签"""
    text = text.replace('\r', '').replace('\n', '')
    text = re.sub(r'<[^>]+>', '', text)  # 去掉HTML标签
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_chinese_text(text):
    """只保留中文（含繁体转简体）、数字和常见中文标点"""
    text = remove_html_and_newlines(text)
    # 转换繁体为简体
    cc = OpenCC('t2s')
    text = cc.convert(text)
    # 保留中文、数字和中文标点
    text = re.sub(r'[^0-9\u4e00-\u9fff，。！？、“”‘’：；（）—…·《》]', '', text)
    return text


def clean_english_text(text):
    """只保留英文、数字和常见英文标点"""
    text = remove_html_and_newlines(text)
    # 保留英文、数字和英文标点
    text = re.sub(r'[^a-zA-Z0-9 ,.!?;:\'"()\[\]\-]', '', text)
    return text

def tokenizer_both(text:str,tokenizer_pre=TOKENIER):
        tokenizer= AutoTokenizer.from_pretrained(tokenizer_pre)
        encoding=tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        return encoding


import random

class DynamicProb:
    def __init__(self, p1=0.6, p2=0.4, min_p1=0.52, min_p2=0.48, delta=0.02):
        self.p1 = p1
        self.p2 = p2
        self.min_p1 = min_p1
        self.min_p2 = min_p2
        self.delta = delta

    def choose(self):
        choice = random.choices(['ch', 'en'], weights=[self.p1, self.p2])[0]
        if choice == 'ch':
            self.p1 = max(self.min_p1, self.p1 - self.delta)
            self.p2 = 1 - self.p1
        else:
            self.p2 = max(self.min_p2, self.p2 - self.delta)
            self.p1 = 1 - self.p2
        return choice, self.p1, self.p2