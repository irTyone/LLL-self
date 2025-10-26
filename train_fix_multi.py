
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
from torch.utils.data import DataLoader 
from model.GPT_self import LLGPT
from utils.utils import tokenizer_both,DynamicProb
from data.datasets import TextDataset,LazyTextDataset
from transformers import AutoTokenizer
from core.config import TOKENIER
import json
from datetime import timedelta
import pickle

config = {
    "emd_dim": 1024,
    "heads": 8,
    "drop_rate": 0.1,
    "vocab_size": 119547
}
WORLD_SIZE = 6  # 使用 GPU 数量（0..WORLD_SIZE-1）

# 数据集文件夹（脚本会从 dataset-ch/ 和 dataset-en/ 读取文件名）
BLOCK_SIZE = 128
STRIDE = 10
BATCH_SIZE = 8
EPOCHS = 100000
LR = 1e-4
GRAD_ACCUM_STEPS = 4
SAVE_PATH = "./checkpoints"
LOG_PATH = "log/train_log.txt"
NUM_WORKERS = 6# DataLoader num_workers
PIN_MEMORY = True


import socket
def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    addr, port = s.getsockname()
    s.close()
    return port
if os.path.exists("dataset_ch_files.pkl"):
    CH = pickle.load(open("dataset_ch_files.pkl", "rb"))
    EH = pickle.load(open("dataset_en_files.pkl", "rb"))
else:
    CH = sorted(os.listdir('dataset-ch'))
    EH = sorted(os.listdir('dataset-en'))
    pickle.dump(CH, open("dataset_ch_files.pkl", "wb"))
    pickle.dump(EH, open("dataset_en_files.pkl", "wb"))

def train_one_epoch(rank, model, dataloader, optimizer, scaler, criterion, scheduler, epoch):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(rank, non_blocking=True)
        target_ids = target_ids.to(rank, non_blocking=True)

        with autocast():
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * GRAD_ACCUM_STEPS

        if rank == 0 and step % 100 == 0:
            print(f"[Epoch {epoch+1}] Step {step}/{len(dataloader)} Loss: {loss.item()*GRAD_ACCUM_STEPS:.4f}")
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"{epoch+1},{step},{loss.item()*GRAD_ACCUM_STEPS:.6f}\n")

    return total_loss / max(1, len(dataloader))


# ========== 训练入口 ==========
def train_ddp(rank, world_size, resume=False):
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, timeout=timedelta(minutes=30))

    device = torch.device(f"cuda:{rank}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIER)
    tokenizer.pad_token = tokenizer.eos_token

    if rank == 0:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("epoch,step,loss\n")
        os.makedirs(SAVE_PATH, exist_ok=True)

    model = LLGPT(config, block_nums=12, is_transformer_block=4).to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000000)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    selector = DynamicProb()
    ch_i, en_i, start_epoch = 0, 0, 0

    # ===== 断点续训 =====
    ckpt_path = os.path.join(SAVE_PATH, "model_latest.pth")
    if resume and os.path.exists(ckpt_path):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        ckpt = torch.load(ckpt_path, map_location=map_location)
        model.module.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        ch_i = ckpt.get("ch_i", 0)
        en_i = ckpt.get("en_i", 0)
        if rank == 0:
            print(f" 恢复训练：epoch={start_epoch}, ch_i={ch_i}, en_i={en_i}")

    # rank0 读取文件名并广播
   
    ch_files = CH
    en_files = EH
   

    for epoch in range(start_epoch, EPOCHS):
        
        if rank == 0:
            subset = selector.choose()[0]
            if subset == 'ch':
                index = ch_i % len(ch_files)
                ch_i += 1
            else:
                index = en_i % len(en_files)
                en_i += 1
            choice = (subset, index)
        else:
            choice = None

        choice_list = [choice]
        dist.broadcast_object_list(choice_list, src=0)
        subset, index = choice_list[0]

        data_path = os.path.join(f"dataset-{subset}", ch_files[index] if subset == 'ch' else en_files[index])
        dist.barrier()

        if rank == 0:
            print(f"[Epoch {epoch+1}] Using {subset} file: {data_path}")

        dataset = TextDataset(data_path, tokenizer, BLOCK_SIZE, STRIDE, lang=subset, file_idx=index)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                                drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        sampler.set_epoch(epoch)

        avg_loss = train_one_epoch(rank, model, dataloader, optimizer, scaler, criterion, scheduler, epoch)

        dist.barrier()
        if rank == 0:
            torch.save({
                'epoch': epoch + 1,
                'ch_i': ch_i,
                'en_i': en_i,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, ckpt_path)
            print(f"[Rank 0] ✅ Checkpoint saved ({epoch+1})  avg_loss={avg_loss:.4f}")

    dist.barrier()
    dist.destroy_process_group()


# ========== 启动 ==========
if __name__ == "__main__":
    port = find_free_port()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    print(f"Using MASTER_ADDR={os.environ['MASTER_ADDR']} MASTER_PORT={os.environ['MASTER_PORT']}")

    mp.spawn(train_ddp, args=(WORLD_SIZE, True), nprocs=WORLD_SIZE)