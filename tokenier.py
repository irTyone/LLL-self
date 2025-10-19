from datasets import load_dataset, DownloadConfig
import json
from itertools import islice
from tqdm import tqdm
from core.config import DATASET, DATA_SAVE,beijingtz,DATA_SAVE_CH,SUBSET_CH
from datetime import  datetime
import os
class Dataloader:
    def __init__(self, dataset, subset, offset=0, max_retries=5):
        self.dataset = dataset
        self.subset = subset
        self.offset = offset
        self.download_config = DownloadConfig(max_retries=max_retries, use_etag=True)

    def data_download(self, stream=True, split="train", batch_size=1000, data_save='',record_json=""):
        count=0
        batch_data=[]
        if self.subset:
            dataset = load_dataset(self.dataset, self.subset, download_config=self.download_config, streaming=stream)[
                split]
        else:
            dataset = load_dataset(self.dataset, download_config=self.download_config, streaming=stream)
        ds = islice(dataset, self.offset, None)
        for i, data in tqdm(enumerate(ds), total=batch_size, desc="数据下载进度"):
            count=i
            if i == batch_size:
                break
            batch_data.append(data)
        if record_json=='':
            with open("record_rows.json",'r',encoding="utf-8")as f:
                old_cursor=json.load(f)
            with open("record_rows.json", 'w', encoding="utf-8")as f:
                old_cursor["offset"]=old_cursor["offset"]+count
                json.dump(old_cursor,f,ensure_ascii=False)
        else:
            with open(record_json, 'r', encoding="utf-8")as f:
                old_cursor = json.load(f)
            with open(record_json, 'w', encoding="utf-8")as f:
                old_cursor["offset"] = old_cursor["offset"] + count
                json.dump(old_cursor, f, ensure_ascii=False)
        if data_save!='':
            os.makedirs(data_save,exist_ok=True)
            with open(f"{data_save}/text_{datetime.now(beijingtz).strftime('%m-%d-%H-%M-%S')}.json", 'w', encoding="utf-8") as f:
                json.dump(batch_data, f, ensure_ascii=False)
        else:
            with open(f"text_{datetime.now(beijingtz).strftime('%m-%d-%H-%M-%S')}.json", 'w', encoding="utf-8") as f:
                json.dump(batch_data, f, ensure_ascii=False)






if __name__ == '__main__':
    with open("record_ch.json",'r',encoding='utf-8') as f:
        cursor=json.load(f)
    for i in range(1000):
        ds = Dataloader("wikimedia/wikipedia", SUBSET_CH,offset=cursor["offset"])
        ds.data_download(data_save=DATA_SAVE_CH,record_json="record_ch.json")
