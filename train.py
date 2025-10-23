import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch
import os
from torch.utils.data import DataLoader 
from model.GPT_self import LLGPT
from utils.utils import tokenizer_both
from data.datasets import TextDataset
from transformers import AutoTokenizer
from core.config import TOKENIER
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
dist.init_process_group(backend='nccl')  # 单机多卡用 nccl
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
config = {
        "emd_dim": 728,
        "heads": 8,
        "drop_rate": 0.1,
        "vocab_size": 119547
    }
def train(model, dataloader, epochs=5, lr=1e-4, device='cuda',
                               gradient_accumulation_steps=2, save_path="./checkpoints"):

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader)*epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            with autocast():
                # print(f"input_ids:{input_ids.shape}")
                input_ids_cuda = input_ids.to('cuda')
                max_id = input_ids_cuda.max()
                min_id = input_ids_cuda.min()

                # print(f"**【最终检查】** Input IDs 最大值: {max_id.item()}")
                # print(f"**【最终检查】** Input IDs 最小值: {min_id.item()}")
                outputs = model(input_ids)  # [B, L, vocab_size]
                loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                loss = loss / gradient_accumulation_steps
                print(f"loss:{loss}")
            scaler.scale(loss).backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * gradient_accumulation_steps

            if step % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{step}/{len(dataloader)}] "
                      f"Loss: {loss.item()*gradient_accumulation_steps:.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")

        # 保存检查点
        checkpoint_path = os.path.join(save_path, f"model_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

def get_dataloader(file_path, tokenizer, block_size=128, stride=1, batch_size=16):
    dataset = TextDataset(file_path, tokenizer, block_size, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def main(file_path="DATASET-ch/text_10-20-18-18-19.json",
         block_size=256,
         stride=10,
         batch_size=32,
         epochs=3,
         lr=1e-4,
         transformer_blocks=4,
         device='cuda',
         gradient_accumulation_steps=4,
         save_path="./checkpoints"):

    
    tokenizer= AutoTokenizer.from_pretrained(TOKENIER)
    tokenizer.pad_token = tokenizer.eos_token  # GPT 风格 padding


    dataloader = get_dataloader(file_path, tokenizer, block_size, stride, batch_size)

    print(tokenizer.vocab_size)


    
    model = LLGPT(config, block_nums=12,is_transformer_block=transformer_blocks)

    
    train(
        model,
        dataloader,
        epochs=epochs,
        lr=lr,
        device=device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_path=save_path
    )


if __name__ == "__main__":
    main()