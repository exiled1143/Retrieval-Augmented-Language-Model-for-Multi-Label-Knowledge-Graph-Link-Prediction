from typing import List
from KnowledgeGraph.utils.log import Log
from argparse import ArgumentParser
import torch
import random
import numpy as np
import json


class GlobalVariable:
    class_num = 57005
    use_special_token_type_embeddings = False 
    use_special_segment_embeddings = False

def set_seed(args: ArgumentParser):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def output_sigmoid_label(idx_list: List[int]) -> List[int]:
    zero_one_label_list = [0] * GlobalVariable.class_num
    for idx in idx_list:
        zero_one_label_list[idx] = 1
    return zero_one_label_list

def collate_fn_for_finetuning(batch_data: List[dict]) -> dict:
    """
    Collate function for finetuning.
    Args:
        batch_data: List[dict]
            mem_mrm_special_token_type_ids: [0(head), 1(relation), 2(tail), 3(description), 4(SEP), 5(CLS), 6(PAD)]
            mem_mrm_token_ids: 101(CLS) 102(SEP) 103(MASK)
            e.g. batch_data = [
                {
                "mem_mrm_token_ids":[101(CLS),103(MASK),234(relation),43(relation),1030(tail), 111(desciption), 102(sep), ...],
                "mem_mrm_target_ids":[20,32,1,5,50000],
                "mem_mrm_special_segment_ids":[0,0,0,0,0,0,0,1,1,1,...],
                "mem_mrm_special_token_type_ids":[5,0,1,1,2,3,4,0,...]},
                {
                "mem_mrm_token_ids":[101(CLS),103(MASK),23(relation),131(tail), 111(desciption), 102(sep), ...],
                "mem_mrm_target_ids":[6,7,51000],
                "mem_mrm_special_segment_ids":[0,0,0,0,0,0,1,1...],
                "mem_mrm_special_token_type_ids":[5,0,1,2,3,4,0,...]},...
            ]
            
    Returns:
        e.g.
        merge_data = {
            "input_ids":[[101,103,234,43,1030,111,102,...],[101,103,23,131,111,102,...]], shape:(batch, max_seq_len)
            "special_token_type_ids":[[5,0,1,1,2,3,4,0,...,6,6,6],[5,0,1,2,3,4,0,...,6,6]], shape:(batch, max_seq_len)
            "special_segment_ids":[[0,0,0,0,0,0,0,1,1,1,..],[0,0,0,0,0,0,1,1,...]], shape:(batch, max_seq_len)
            "task_direction":[1,1], shape:(batch)
            "labels":[[0,1,0,0,0,1,0,0,0,0,...1,0,..],[0,0,0,0,0,0,1,1,0,0,...,1,0,..]], shape:(batch, mlm_mem_mrm_class_num)
            "attention_mask":[[1,1,1,1,..,0,0,0],[1,1,1,1,...,1,0,0]] shape:(batch, max_seq_len)
        }
    """

    special_token_type_id_for_pad = 6
    token_id_for_pad = 0
    merge_data = {
        "input_ids":[],
        "special_token_type_ids":[],
        "special_segment_ids":[],
        "task_direction":[],
        "labels":[],
        "attention_mask":[],
    }
    collect_max_seq_len = []
    try:
        collect_max_seq_len.append(max([len(each_data["mem_mrm_token_ids"]) for each_data in batch_data]))
    except:
        Log.info("no mem_mrm_token_ids")
    
    max_seq_len = max(collect_max_seq_len)
    assert max_seq_len <= 512
    
    for each_data in batch_data:
        if "mem_mrm_token_ids" in each_data:
            mem_mrm_token_ids  = [each_data["mem_mrm_token_ids"] + [token_id_for_pad] * (max_seq_len - len(each_data["mem_mrm_token_ids"]))] #(batch,max_seq_len)
            mem_mrm_target_ids = [output_sigmoid_label(each_data["mem_mrm_target_ids"])]
            mem_mrm_special_segment_ids    = [each_data["mem_mrm_special_segment_ids"] + [max(each_data["mem_mrm_special_segment_ids"])+1] * (max_seq_len - len(each_data["mem_mrm_special_segment_ids"]))] #(batch,max_seq_len)
            mem_mrm_special_token_type_ids = [each_data["mem_mrm_special_token_type_ids"] + [special_token_type_id_for_pad] * (max_seq_len - len(each_data["mem_mrm_special_token_type_ids"]))] #(batch,max_seq_len)
            mem_mrm_attention_mask         = [[1.0] * len(each_data["mem_mrm_token_ids"]) + [0.0] * (max_seq_len - len(each_data["mem_mrm_token_ids"]))]
            merge_data["input_ids"].extend(mem_mrm_token_ids)
            merge_data["labels"].extend(mem_mrm_target_ids)
            merge_data["special_segment_ids"].extend(mem_mrm_special_segment_ids)
            merge_data["special_token_type_ids"].extend(mem_mrm_special_token_type_ids)
            merge_data["attention_mask"].extend(mem_mrm_attention_mask)
            merge_data["task_direction"].extend([1])

    merge_data["input_ids"] = torch.tensor(merge_data["input_ids"], dtype=torch.long)
    merge_data["special_token_type_ids"] = torch.tensor(merge_data["special_token_type_ids"], dtype=torch.long)
    merge_data["special_segment_ids"] = torch.tensor(merge_data["special_segment_ids"], dtype=torch.long)
    merge_data["task_direction"] = torch.tensor(merge_data["task_direction"], dtype=torch.long)
    merge_data["labels"] = torch.tensor(merge_data["labels"], dtype=torch.long)
    merge_data["attention_mask"] = torch.tensor(merge_data["attention_mask"], dtype=torch.float)
    return merge_data

def build_train_data(file_paths:str) -> List[dict]:
    collect_data = {} #dict of list of dict
    for file_path in file_paths:
        with open(file_path, "r") as fp:
            data = json.load(fp)
            print(len(data))
            if ('mem' in file_path) or ('mrm' in file_path):
                if 'mem_mrm' in collect_data:
                    collect_data['mem_mrm'] = collect_data['mem_mrm'] + data
                else:
                    collect_data['mem_mrm'] = data

    max_len_data = 0
    for k, v in collect_data.items():
        max_len_data = max(max_len_data, len(v))

    result_data  = {}
    for key, value in collect_data.items():
        padding_data_num = max_len_data-len(value)
        padding_data = random.choices(value , k = padding_data_num)
        result_data[key] = value + padding_data
        random.shuffle(result_data[key])
    
    output_data = []
    for i in range(max_len_data): 
        combine = {}
        for key in result_data.keys():
            combine.update(result_data[key][i])
        output_data.append(combine)
    print("total pages",len(output_data))
    return output_data


