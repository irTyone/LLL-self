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
from utils.utils import tokenizer_both
from data.datasets import TextDataset
from transformers import AutoTokenizer
from core.config import TOKENIER


config = {
    "emd_dim": 728,
    "heads": 8,
    "drop_rate": 0.1,
    "vocab_size": 119547
}
WORLD_SIZE = 6  # 使用 6 块 GPU
TOKENIZER = "gpt2"  # 替换为你的 tokenizer
FILE_PATH = "DATASET-ch/text_10-20-18-18-19.json"
BLOCK_SIZE = 256
STRIDE = 10
BATCH_SIZE = 32
EPOCHS = 3
LR = 1e-4
GRAD_ACCUM_STEPS = 4
SAVE_PATH = "./checkpoints"


def train_ddp(rank):
    # 设置 GPU 和分布式环境
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=WORLD_SIZE)
    device = torch.device(f"cuda:{rank}")

    # Tokenizer & Dataloader
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = TextDataset(FILE_PATH, tokenizer, BLOCK_SIZE, STRIDE)
    sampler = DistributedSampler(dataset, num_replicas=WORLD_SIZE, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    # 模型
    model = LLGPT(config, block_nums=12, is_transformer_block=4).to(device)
    model = DDP(model, device_ids=[rank])

    # 原 train 内部逻辑几乎不变
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader)*EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    os.makedirs(SAVE_PATH, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0
        optimizer.zero_grad()

        for step, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            with autocast():
                outputs = model(input_ids)
                loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                loss = loss / GRAD_ACCUM_STEPS
                print(f"loss:{loss}")
            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * GRAD_ACCUM_STEPS

            if step % 100 == 0 and rank == 0:
                print(f"[Epoch {epoch+1}/{EPOCHS}] Step {step}/{len(dataloader)} "
                      f"Loss: {loss.item()*GRAD_ACCUM_STEPS:.4f}")

        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            print(f"[Epoch {epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}")
            checkpoint_path = os.path.join(SAVE_PATH, f"model_epoch{epoch+1}.pt")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(train_ddp, nprocs=WORLD_SIZE, args=())