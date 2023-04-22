from KnowledgeGraph.application.graph_bert_finetune import GraphBertFinetune
from KnowledgeGraph.utils.utils import build_train_data, collate_fn_for_finetuning
from KnowledgeGraph.utils.utils import GlobalVariable
from KnowledgeGraph.metrics.precision_at_k import PrecisionAtKTotal
from KnowledgeGraph.metrics import Metrics
from KnowledgeGraph.utils.log import Log
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers import AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from typing import NoReturn, List
from tqdm import tqdm
import torch
Log.show_ddp_info = False


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="./data", nargs="+", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--mem_mrm_class_num", default=57005, type=int, help="total number of classes of entites and relations")
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--restore_weight", default="", type=str)
    parser.add_argument("--precision_at_k", default=1, nargs="+", type=int)
    parser.add_argument("--focal_gamma", default=0., type=float, help="gamma=0: same as cross entropy, gamma實驗1~5之間，越大可以減少簡單資料的loss值")
    parser.add_argument("--add_precision_in_loss", default="none", type=str, help="none or multiply")
   
    # Args
    args = parser.parse_args()
    args.fusion_cls_in_mem_mrm = False
    args.pos_ratio = None
    args.neg_ratio = None
    print("="*10)
    print(f"show parameters:{args}")

    # update class num
    GlobalVariable.class_num = args.mem_mrm_class_num

    # bert config
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    print("="*10)
    print(f"show bert pretrained model:{config}")
    
    # build data
    data = build_train_data(args.data_dir)
    
    # build model
    graph_finetune_model = GraphBertFinetune(args=args, config=config, tokenizer=tokenizer)
    print("="*10)
    print(f"build graph finetune model successfully.")
    graph_finetune_model.model.resize_token_embeddings(args.mem_mrm_class_num)

    # restore model
    if args.restore_weight:
        try:
            graph_finetune_model.load_state_dict(torch.load(args.restore_weight))
        except RuntimeError:
            graph_finetune_model.load_state_dict(torch.load(args.restore_weight, map_location=torch.device('cpu')))
        print("="*10)
        print(f"restore graph finetune model weight successfully.")
    device = torch.device("cuda", 0) if torch.cuda.is_available() else "cpu" 
    graph_finetune_model.to(device)

    # predict
    print("="*10)
    print(f"model predict")
    graph_finetune_model.eval()
    register_metrics = [PrecisionAtKTotal(topN=i) for i in args.precision_at_k]
    predict_topK(graph_finetune_model, test_data=data, register_metrics=register_metrics, args=args, device=device)

def predict_topK(model: BertPreTrainedModel, test_data: List[dict], register_metrics: List[Metrics], args, device) -> NoReturn:
    dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_for_finetuning, drop_last=False)
    print(f"total nums:{len(dataloader)}")
    for datum in dataloader:
        datum = {key: value.to(device) for key, value in datum.items()}
        GT = datum["labels"] # GT shape: (batch, class_num)
        datum["labels"] = None
        _, mem_mrm_out, _, _, _ = model(**datum) # mem_mrm_out shape:(batch, class_num)
        if mem_mrm_out != None:
            
            # get bigger than zero data(use in recall, precision)
            over_zero = mem_mrm_out > 0
            
            # sort the indices which over zero
            sort_mem_mrm_out = torch.argsort(mem_mrm_out, descending=True, dim=1) # sort_mem_mrm_out shape:(batch, class_num)

            over_zero_nums = torch.sum(over_zero, axis=-1) # over_zero_nums shape: (batch)
            for over_zero_num,sort_mem_mrm_out_datum, each_gt in zip(over_zero_nums, sort_mem_mrm_out, GT):
                over_zero_num = over_zero_num.cpu().numpy()
                sort_mem_mrm_out_datum=sort_mem_mrm_out_datum.cpu().numpy().tolist()
                prediction_indices = sort_mem_mrm_out_datum[:over_zero_num] # prediction_indices shape: List[int]
                gt = torch.squeeze(each_gt.nonzero()).cpu().numpy().tolist() # gt shape: List[int]
                if type(gt) == int:
                    gt = [gt]
                if type(prediction_indices) != list:
                    prediction_indices = [prediction_indices]
                for register_metric in register_metrics:
                    register_metric.calculate(targets=gt, predictions=prediction_indices)
    for register_metric in register_metrics:
        register_metric.show_result()

if __name__ == "__main__":
    main()
