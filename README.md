
# Retrieval-Augmented-Language-Model-for-Multi-Label-Knowledge-Graph-Link-Prediction

This repository is the official implementation of **Retrieval-Augmented-Language-Model-for-Multi-Label-Knowledge-Graph-Link-Prediction**

## Requirements

python version:
- python3.6.9

python package:
- apex         == 0.9.10.dev0
- torch        == 1.9.0+cu102
- tqdm         == 4.53.0
- transformers == 4.18.0
- wandb        == 0.13.10

Install requirements:

- Install KnowledgeGraph package
```
pip install -e .
```
- Install related package
```setup
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Dataset
- WN18RR
- FB15K-237

unzip dataset/FB15k-237/fb15k237.zip and dataset/WN18RR/wn18rr.zip

Our proposed description dataset for FB15k237 is at dataset/FB15k-237/entity2text.csv

The license of the proposed dataset is in CC-BY-2.5.txt

## Training

To train the model in the paper, run this command:
- WN18RR

stage1(4 GPU cards)
```
TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 model_finetune_ddp.py --batch_size 20 --data_dir dataset/WN18RR/fine_tune_mem_mask_tail.json dataset/WN18RR/fine_tune_mem_mask_head.json --optimizer adam --warmup_steps 12000  --save_interval 1 --storyid WN18RR_finetune_stage1 --use_amp --gradient_accumulation_steps 2 --lr_divide_scalar 2 --mem_mrm_class_num 57005 --positive_ratio 30000 
```
stage2(4 GPU cards)
```
TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 model_finetune_ddp.py --batch_size 20 --data_dir dataset/WN18RR/fine_tune_mem_mask_tail.json dataset/WN18RR/fine_tune_mem_mask_head.json --optimizer adam --warmup_steps 12000 --save_interval 1 --storyid WN18RR_finetune_stage2 --use_amp --gradient_accumulation_steps 2 --lr_divide_scalar 2 --mem_mrm_class_num 57005 --positive_ratio 100 --add_precision_in_loss multiply  --restore_weight KnowledgeGraph/weight/WN18RR_finetune_stage1/31.pt
```
stage3(4 GPU cards)
```
TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 model_finetune_ddp.py --batch_size 20 --data_dir dataset/WN18RR/fine_tune_mem_mask_tail.json dataset/WN18RR/fine_tune_mem_mask_head.json --optimizer adam --warmup_steps 12000  --save_interval 1 --storyid WN18RR_finetune_stage3 --use_amp --gradient_accumulation_steps 2 --lr_divide_scalar 2 --mem_mrm_class_num 57005 --positive_ratio 2 --add_precision_in_loss multiply  --restore_weight KnowledgeGraph/weight/WN18RR_finetune_stage2/9.pt
```

- FB15K-237

stage1(8 GPU cards)
```train
TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 model_finetune_ddp.py --batch_size 10 --data_dir dataset/FB15k-237/top_3_fine_tune_mem_mask_tail.json dataset/FB15k-237/top_3_fine_tune_mem_mask_head.json --optimizer adam --warmup_steps 12000  --save_interval 1 --storyid fb15k237_finetune_stage1 --use_amp --gradient_accumulation_steps 4 --lr_divide_scalar 2 --mem_mrm_class_num 42516 --positive_ratio 30000 --project_name fb15k237
```

stage2(4 GPU cards)
```
TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 model_finetune_ddp.py --batch_size 10 --data_dir dataset/FB15k-237/top_3_fine_tune_mem_mask_tail.json dataset/FB15k-237/top_3_fine_tune_mem_mask_head.json --optimizer adam --warmup_steps 12000 --save_interval 1 --storyid fb15k237_finetune_stage2--use_amp --gradient_accumulation_steps 4 --lr_divide_scalar 2 --mem_mrm_class_num 42516 --positive_ratio 20  --project_name fb15k237 --add_precision_in_loss multiply --restore_weight KnowledgeGraph/weight/fb15k237_finetune_stage1/42.pt
```

stage3(4 GPU cards)
```
TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 model_finetune_ddp.py --batch_size 10 --data_dir dataset/FB15k-237/top_3_fine_tune_mem_mask_tail.json dataset/FB15k-237/top_3_fine_tune_mem_mask_head.json --optimizer adam --warmup_steps 12000 --save_interval 1 --storyid fb15k237_finetune_stage3 --use_amp --gradient_accumulation_steps 4 --lr_divide_scalar 2 --mem_mrm_class_num 42516 --positive_ratio 2 --project_name fb15k237 --add_precision_in_loss multiply --restore_weight KnowledgeGraph/weight/fb15k237_finetune_stage2/7.pt
```


## Evaluation

- WN18RR
```eval
python model_eval.py --data_dir dataset/WN18RR/test_mem_mask_head.json  dataset/WN18RR/test_mem_mask_tail.json --restore_weight KnowledgeGraph/weight/WN18RR_finetune/best_weight.pt --precision_at_k 1 3 5 --batch_size 10 --mem_mrm_class_num 57005
```
You can replace  **--restore_weight**  with the weight that you trained on stage3.

- FB15K-237
```eval
python model_eval.py --data_dir dataset/FB15k-237/top_3_test_mem_mask_head.json  dataset/FB15k-237/top_3_test_mem_mask_tail.json  --batch_size 10 --precision_at_k 1 3 5 --mem_mrm_class_num 42516 --restore_weight KnowledgeGraph/weight/FB15K-237_finetune/best_weight.pt
```
You can replace  **--restore_weight**  with the weight that you trained on stage3.


## Pre-trained Models
We used bert-base-cased pretrained model which loaded from [huggingface](https://huggingface.co/bert-base-cased).



## Results

Our model achieves the following performance on :

### **Precision@K on WN18RR**

| Model name         | Precision@1  | Precision@3 | Precision@5 |
| ------------------ |---------------- | -------------- | -------------- |
| Our Model   |     35.43%         |      26.24%       | 23.27% |

### **Precision@K on FB15K-237**

| Model name         | Precision@1  | Precision@3 | Precision@5 |
| ------------------ |---------------- | -------------- | -------------- |
| Our Model  |     32.25%         |      22.7%       | 19.79% |

