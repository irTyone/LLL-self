import os
import json
from utils.utils import clean_chinese_text, clean_english_text  # 替换为实际导入路径

# 文件夹路径
folders = {
    'ch': 'dataset-ch',
    'en': 'dataset-en'
}

for lang, folder in folders.items():
    if not os.path.isdir(folder):
        print(f"文件夹 {folder} 不存在，跳过")
        continue

    files = [f for f in os.listdir(folder) if f.endswith('.json')]
    for fname in files:
        file_path = os.path.join(folder, fname)
        print(f"正在处理 {file_path} ...")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 清洗 text 字段
        for item in data:
            if 'text' in item:
                if lang == 'ch':
                    item['text'] = clean_chinese_text(item['text'])
                else:
                    item['text'] = clean_english_text(item['text'])

        # 覆盖写回原文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"{file_path} 已完成清洗并覆盖")