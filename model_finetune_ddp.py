from KnowledgeGraph.application.graph_bert_finetune import GraphBertFinetune
from KnowledgeGraph.utils.utils import build_train_data, set_seed, collate_fn_for_finetuning
from KnowledgeGraph.utils.utils import GlobalVariable
from KnowledgeGraph.utils.io import check_dir
from KnowledgeGraph.learning_rate.attention_learning_rate import TransformerWarmup
from KnowledgeGraph.utils.log import Log
from argparse import ArgumentParser
from torch.optim import lr_scheduler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataset import Dataset
from torch.optim import Adam, AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import AutoConfig
from transformers import AutoTokenizer
from functools import reduce
import torch
import torch.distributed as dist
import os
import time
import wandb


Log.show_ddp_info = False
class CustomDataset(Dataset):
    def __init__(self, origin_data):
        self.origin_data = origin_data

    def __len__(self):
        return len(self.origin_data)

    def __getitem__(self, idx):
        return self.origin_data[idx]

def main():

    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="./data", nargs="+", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--mem_mrm_class_num", default=57005, type=int, help="total number of classes of entites and relations")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size used by each gpu")
    parser.add_argument("--num_train_epochs", default=100, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--warmup_ratio", default=0.2, type=float)
    parser.add_argument("--warmup_steps", default=12000, type=int)
    parser.add_argument("--lr_divide_scalar", default=2., type=float, help="")
    parser.add_argument("--eps", default=1e-9, type=int)
    parser.add_argument('--use_amp', dest='use_amp', action='store_true', default=False)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--local_rank", default=0, type=int, help="use default value")
    parser.add_argument("--mlm_loss_ratio", default=1.0, type=float)
    parser.add_argument("--mem_mrm_loss_ratio", default=1.0, type=float)
    parser.add_argument("--save_interval", default=1, type=int)
    parser.add_argument("--storyid", default="KG1", type=str, help="experiment name in wandb")
    parser.add_argument("--save_path", default="KnowledgeGraph/weight", type=str)
    parser.add_argument("--restore_weight", default="", type=str)
    parser.add_argument("--project_name", default="KG", type=str, help="project name in wandb")
    parser.add_argument("--seed", default=1231, type=int, help="set weight seed")
    parser.add_argument("--mem_loss", default="sigmoid", type=str, help="softmax(ce_loss) or sigmoid(customize multi_label_loss)")
    parser.add_argument("--focal_gamma", default=0., type=float, help="gamma=0: same as cross entropy")
    parser.add_argument("--add_precision_in_loss", default="none", type=str, help="none or multiply")
    parser.add_argument("--positive_ratio", default=1.0, type=float, help="loss positive ratio")
    parser.add_argument("--negative_ratio", default=1.0, type=float, help="loss negative ratio")
    args = parser.parse_args()
    args.fusion_cls_in_mem_mrm = False
    print(args)

    set_seed(args)

    GlobalVariable.class_num = args.mem_mrm_class_num

    #------------------
    # wandb 
    #------------------
    if args.local_rank == 0:
        run = wandb.init(project=args.project_name)
        run.name = args.storyid

    #------------------
    # DDP
    #------------------
    dist.init_process_group(backend='nccl')
    dist.barrier() 
    world_size = dist.get_world_size()
    args.world_size = world_size

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config.use_special_token_type_embeddings = False
    config.use_special_segment_embeddings = False
    print(config)

    #------------------
    # Obtain positive ratio and negative ratio used in loss function.
    #------------------
    train_data = build_train_data(args.data_dir)
    if args.positive_ratio != 1.0:
        args.pos_ratio = args.positive_ratio
    else:
        target_ratio = len(reduce(lambda x,y: x+y["mem_mrm_target_ids"], train_data, []))/(len(train_data)*(args.mem_mrm_class_num)) #平均每筆資料佔用的pos數量
        args.pos_ratio = args.positive_ratio if args.positive_ratio != 1.0 else 1/target_ratio
    
    args.neg_ratio = args.negative_ratio if args.negative_ratio != 1.0 else 1.0
    print("pos_ratio",args.pos_ratio, "neg_ratio", args.neg_ratio, "add_precision_in_loss", args.add_precision_in_loss)
    if args.local_rank == 0:
        wandb.config.update(args)

    graph_finetune_model = GraphBertFinetune(args=args, config=config, tokenizer=tokenizer)
    graph_finetune_model.model.resize_token_embeddings(args.mem_mrm_class_num)
    
    if args.restore_weight and dist.get_rank() == 0:
        try:
            graph_finetune_model.load_state_dict(torch.load(args.restore_weight))
            print("="*10)
            print(f"restore graph pretrain model weight successfully")
        except RuntimeError:
            print("cannot restore graph pretrain model weight, use initail weight")

    device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else "cpu" 
    graph_finetune_model.to(device)


    graph_finetune_model = torch.nn.parallel.DistributedDataParallel(graph_finetune_model, device_ids=[args.local_rank],
                                                                     output_device=args.local_rank, find_unused_parameters=True)


    train_dataset = CustomDataset(origin_data=train_data)
    #------------------
    # DDP sampler
    #------------------
    train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                              pin_memory=True, prefetch_factor=2, num_workers=4, collate_fn=collate_fn_for_finetuning, drop_last=True)


    #------------------
    # DDP training
    #------------------
    if args.use_amp:
        train_amp(model=graph_finetune_model, train_dataloader=train_loader, args=args, device=device, train_sampler=train_sampler)    

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def get_optimizer_schedule(optimizer, warmup_steps: int=4000, lr_divide_scalar: float=1.0):
    lr_scalar = 1./lr_divide_scalar
    lambda2 = TransformerWarmup(scalar=lr_scalar, warmup_steps=warmup_steps)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
    return scheduler

def save_weight(path, weights):
    check_dir(path)
    torch.save(weights, path)

def train_amp(model, train_dataloader: DataLoader, args: ArgumentParser, device, train_sampler) -> None:
    """
    training AMP pipeline
    """

    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = args.warmup_steps or int(total_steps * args.warmup_ratio)
    if args.optimizer=="adam":
        optimizer = Adam(model.parameters(), lr=1., eps=args.eps)
    elif args.optimizer=="adamw":
        optimizer = AdamW(model.parameters(), lr=1., weight_decay=0.05)
    

    scheduler = get_optimizer_schedule(optimizer=optimizer, warmup_steps=warmup_steps, lr_divide_scalar=args.lr_divide_scalar)

    scaler = GradScaler()
    num_steps = 0
    wandb_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        train_sampler.set_epoch(epoch)
        print("epoch",epoch, "total steps:", len(train_dataloader))
        collect_loss = []
        collect_mem_mrm_loss = []
        accumulate_loss,  accumulate_mem_mrm_loss = torch.tensor(0., device=args.local_rank), torch.tensor(0., device=args.local_rank)
        
        start_time = time.time()
        optimizer.zero_grad()
        for step, batch in enumerate(train_dataloader):
            if step % 100 == 0 and args.local_rank == 0:
                interval = round((time.time()-start_time)/100,3)
                print("="*10)
                print(f"one batch used time(s):{interval}")
                print(f"residual time in one epoch(s):{(len(train_dataloader)-step)*interval}")
                print(f"residual training time(s):{(len(train_dataloader)-step)*interval+(args.num_train_epochs-epoch-1)*interval*len(train_dataloader)}")
                print("="*10)
                start_time = time.time()
            model.train()
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            
            # AMP
            with autocast(): 
                outputs = model(**batch)
                Log.ddp_info(f"==local_rank:{args.local_rank}=====loss-info:total_loss:{outputs[0]},{outputs[-2]}====")
            loss = outputs[0] / args.gradient_accumulation_steps

            check_loss         = reduce_mean(loss, dist.get_world_size())
            try:
                check_mem_mrm_loss = reduce_mean(outputs[-2]["mem_mrm_loss"]/args.gradient_accumulation_steps, dist.get_world_size())
            except TypeError:
                check_mem_mrm_loss = 0

            if args.local_rank == 0:
                accumulate_loss             += check_loss
                accumulate_mem_mrm_loss     += check_mem_mrm_loss

            if step % 100 == 0 and args.local_rank == 0:
                wandb_steps += 1
                lr = scheduler.get_lr()
                wandb.log({'learning_rate': lr[0]},step=wandb_steps)
                wandb.log({'loss(ratio)': check_loss*args.gradient_accumulation_steps, 
                           'mem_mrm loss(no ratio)': check_mem_mrm_loss*args.gradient_accumulation_steps}, step=wandb_steps) 
            
            scaler.scale(loss).backward()


            if (step+1) % args.gradient_accumulation_steps == 0:
                # record loss
                if args.local_rank == 0:
                    collect_loss.append(accumulate_loss)
                    collect_mem_mrm_loss.append(accumulate_mem_mrm_loss)
                    accumulate_loss, accumulate_mem_mrm_loss = torch.tensor(0., device=args.local_rank), torch.tensor(0., device=args.local_rank)
                num_steps += 1
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                
                scheduler.step()
                model.zero_grad()
                lr = scheduler.get_lr()
                Log.ddp_info(f"==local_rank:{args.local_rank}=====lr:{lr}====")

        try:
            if args.local_rank == 0:
                print("epoch avg loss", sum(collect_loss)/len(collect_loss))
                wandb.log({'epoch avg loss(ratio)': sum(collect_loss)/len(collect_loss),
                           'epoch avg mem_mrm loss(no ratio)': sum(collect_mem_mrm_loss)/len(collect_mem_mrm_loss)}, step=epoch)
        except:
            print("cannot log")

        # save model weights
        if (epoch+1) % args.save_interval == 0 and dist.get_rank() == 0:
            save_weight(path=f"{args.save_path}{os.sep}{args.storyid}{os.sep}{epoch+1}.pt", weights=model.module.state_dict())

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


if __name__ == "__main__":
    main()