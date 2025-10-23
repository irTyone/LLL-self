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
from data.datasets import TextDataset
from transformers import AutoTokenizer
from core.config import TOKENIER

config = {
    "emd_dim": 1024,
    "heads": 8,
    "drop_rate": 0.1,
    "vocab_size": 119547
}
WORLD_SIZE = 6  # 使用 6 块 GPU

FILE_PATH = "DATASET-ch/text_10-20-18-18-19.json"
BLOCK_SIZE = 256
STRIDE = 10
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-4
GRAD_ACCUM_STEPS = 4
SAVE_PATH = "./checkpoints"
LOG_PATH = "log/train_log.txt"  

# ========== 主训练函数 ==========
def train_one_epoch(rank, model, dataloader, optimizer, scaler, criterion, scheduler, epoch):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(rank)
        target_ids = target_ids.to(rank)

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

        # rank 0 写 log
        if rank == 0:
            batch_mean = input_ids.float().mean().item()
            batch_max = input_ids.max().item()
            batch_min = input_ids.min().item()
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"{epoch+1},{step},{loss.item()*GRAD_ACCUM_STEPS:.6f},{batch_mean:.2f},{batch_max},{batch_min}\n")
            if step % 100 == 0:
                print(f"[Epoch {epoch+1}] Step {step}/{len(dataloader)} Loss: {loss.item()*GRAD_ACCUM_STEPS:.4f}")

    return total_loss / len(dataloader)


# ===== 每个 rank 初始化函数 =====
def train_ddp(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=WORLD_SIZE)
    torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIER)
    tokenizer.pad_token = tokenizer.eos_token

    # 初始化 log 文件（只在 rank 0）
    if rank == 0:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("epoch,step,loss,batch_mean,batch_max,batch_min\n")
        os.makedirs(SAVE_PATH, exist_ok=True)

    # 模型、DDP、优化器等（每个 rank 内部初始化）
    model = LLGPT(config, block_nums=12, is_transformer_block=4).to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)  # T_max 之后可以动态调整
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # 动态概率选择
    selector = DynamicProb()
    ch_i, en_i = 0, 0

    for epoch in range(EPOCHS):
        # 选择数据集
        subset, p1, p2 = selector.choose()
        print(f"[Rank {rank}] Epoch {epoch+1} 选择数据集: {subset} 概率: {p1:.2f}:{p2:.2f}")

        if subset == 'ch':
            data_path = f"dataset-ch/{os.listdir('dataset-ch')[ch_i]}"
            ch_i += 1
        else:
            data_path = f"dataset-en/{os.listdir('dataset-en')[en_i]}"
            en_i += 1

        dataset = TextDataset(data_path, tokenizer, BLOCK_SIZE, STRIDE)
        sampler = DistributedSampler(dataset, num_replicas=WORLD_SIZE, rank=rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

        avg_loss = train_one_epoch(rank, model, dataloader, optimizer, scaler, criterion, scheduler, epoch)

        # rank 0 保存 checkpoint
        if rank == 0:
            checkpoint_path = os.path.join(SAVE_PATH, "model_latest.pth")  
            torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path) 
    dist.destroy_process_group()


# ===== 启动多卡训练 =====
if __name__ == "__main__":
    mp.spawn(train_ddp, nprocs=WORLD_SIZE)