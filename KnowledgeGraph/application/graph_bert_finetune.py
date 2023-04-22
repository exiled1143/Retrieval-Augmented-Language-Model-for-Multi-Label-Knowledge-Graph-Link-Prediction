from KnowledgeGraph.model.bert_model import BertModelV2
from KnowledgeGraph.utils.log import Log
from KnowledgeGraph.loss.multi_label_loss import MultiLabelLoss
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertLMPredictionHead
from torch import nn
from torch.nn import functional as F
from typing import Optional
import torch
Log.show_debug = False


class GraphBertFinetune(BertPreTrainedModel):
    """Construct the Bert model and add new classifiers in it."""

    def __init__(self, args, config, tokenizer):
        super().__init__(config)
        self.args  = args
        self.tokenizer = tokenizer
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        self.model = BertModelV2.from_pretrained(args.model_name_or_path)
        self.mlm_classifier     = BertLMPredictionHead(config=config)
        self.mem_mrm_classifier = nn.Linear(config.hidden_size, args.mem_mrm_class_num)
        self.ce_loss = MultiLabelLoss(pos_ratio=args.pos_ratio, neg_ratio=args.neg_ratio, gamma=args.focal_gamma, add_precision_in_loss=args.add_precision_in_loss)

    def forward(self, input_ids: torch.Tensor, special_token_type_ids: torch.Tensor, special_segment_ids: torch.Tensor, attention_mask: torch.Tensor, task_direction: torch.Tensor, labels: Optional[torch.Tensor]=None):
        """
        Args:
            input_ids: token id 
                e.g. [[101,103,234,43,1030,111,102,...],[101,103,23,131,111,102,...]]
                shape:(batch, max_seq_len)
            special_token_type_ids: 0(head), 1(relation), 2(tail), 3(description), 4(SEP), 5(CLS), 6(PAD)
                e.g. [[5,0,1,1,2,3,4,0,...,6,6,6],[5,0,1,2,3,4,0,...,6,6]] 
                shape:(batch, max_seq_len)
            special_segment_ids: 
                e.g. [[0,0,0,0,0,0,0,1,1,1,..],[0,0,0,0,0,0,1,1,...]] 
                shape:(batch, max_seq_len)
            attention_mask: PAD use zero, the others use one.
                e.g. [[1,1,1,1,..,0,0,0],[1,1,1,1,...,1,0,0]], 
                shape:(batch, max_seq_len)
            task_direction:
                e.g. [1,1,...]                
                shape:(batch,)
            labels:
                e.g. [[0,1,0,0,0,1,0,0,0,0,...1,0,..],[0,0,0,0,0,0,1,1,0,0,...,1,0,..]] 
                shape:(batch, mlm_mem_mrm_class_num)
        """

        out = self.model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         special_token_type_ids=special_token_type_ids,
                         special_segment_ids=special_segment_ids) #out:(batch, seq_len, dim)
        last_hidden = out.last_hidden_state #last_hidden: (batch, seq_len, dim)
        
        one_hot = F.one_hot(task_direction, num_classes=max(task_direction)+1)
        task_distribution = torch.sum(one_hot, axis=0)

        total_loss  = 0
        mem_mrm_out = None
        mem_mrm_labels   = None
        mem_mrm_loss     = None

        if 1 in task_direction:
            """
            MEM_MRM
            """

            mem_mrm_out_position = torch.sum(task_distribution[:2])
            mem_mrm_inputs       = torch.index_select(last_hidden, 0, index=torch.tensor([i for i in range(torch.sum(task_distribution[:1]), mem_mrm_out_position)]).to(task_distribution.device))
            
            # concat CLS
            if self.args.fusion_cls_in_mem_mrm == True:
                #print("==========activate CLS token concat==========")
                CLS_embeddings = mem_mrm_inputs[:, 0, :] #(batch, dim)
                CLS_embeddings = CLS_embeddings[:, None, :]  #(batch, 1, dim)
                CLS_embeddings = CLS_embeddings.expand(CLS_embeddings.shape[0], mem_mrm_inputs.shape[1], CLS_embeddings.shape[2]) #(batch, seq_len, dim)
                mem_mrm_inputs = torch.cat([mem_mrm_inputs, CLS_embeddings], axis=-1) # (batch, seq_len, dim) -> (batch, seq_len, dim*2)

            mem_mrm_out  = self.mem_mrm_classifier(mem_mrm_inputs) #mem_mrm_out:(mem_mrm_num, seq_len, mem_mrm_class_num)

            # flattern the tensor
            mask_extraction = (input_ids==self.mask_token_id).float()[:,:,None] #(batch, seq_len, 1)
            mem_mrm_out = mem_mrm_out * mask_extraction
            mem_mrm_out = torch.sum(mem_mrm_out, axis=1) #(batch, mem_mrm_class_num)
            
            if labels != None:
                mem_mrm_labels   = torch.index_select(labels, 0, index=torch.tensor([i for i in range(torch.sum(task_distribution[:1]), mem_mrm_out_position)]).to(task_distribution.device))
                Log.debug("mem_mrm_labels",mem_mrm_labels) # (batch, vocab_size) e.g. [[1,0,1,...],[0,0,1,...],...]

                mem_mrm_loss = self.ce_loss(mem_mrm_out, mem_mrm_labels) #mlm_out:(mem_mrm_num, vocab_size)  mem_mrm_labels: (mem_mrm_num, vocab_size)
                total_loss += (mem_mrm_loss * self.args.mem_mrm_loss_ratio)
                Log.debug("mem_mrm_loss",mem_mrm_loss)

        return total_loss, mem_mrm_out, None, {"mem_mrm_loss":mem_mrm_loss}, {"mem_mrm":{"label":mem_mrm_labels,"prediction":mem_mrm_out}}