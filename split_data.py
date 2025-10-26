import os
import json
from tqdm import tqdm  # 进度条库

# 原始数据集文件夹
folders = ["dataset-ch", "dataset-en"]

for folder in folders:
    if not os.path.isdir(folder):
        print(f"❌ 文件夹 {folder} 不存在，跳过")
        continue

    # 输出文件夹，例如 dataset-ch-split / dataset-en-split
    out_root = f"{folder}-split"
    os.makedirs(out_root, exist_ok=True)

    # 全局编号计数器
    global_idx = 0

    # 遍历每个 JSON 文件
    files = sorted([f for f in os.listdir(folder) if f.endswith('.json')])
    for fname in files:
        file_path = os.path.join(folder, fname)
        print(f"📂 正在处理 {file_path} ...")

        # 读取原始 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data_list = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ {file_path} 不是合法 JSON，跳过")
                continue

        # 拆分保存每条记录到主输出文件夹
        for item in tqdm(data_list, desc=f"拆分 {fname}"):
            out_file = os.path.join(out_root, f"{global_idx}.json")
            with open(out_file, 'w', encoding='utf-8') as f_out:
                json.dump(item, f_out, ensure_ascii=False, indent=2)
            global_idx += 1

        print(f"✅ {file_path} 已拆分成 {len(data_list)} 个文件，当前全局编号到 {global_idx-1}")

print("\n🎉 所有文件拆分完成")