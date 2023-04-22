from transformers.models.bert.modeling_bert import BertEmbeddings
from torch import nn
from KnowledgeGraph.utils.utils import GlobalVariable
import torch


class BertEmbeddingsV2(BertEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings."""
    
    def __init__(self, config):
        super().__init__(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        max_special_token_type_ids = 7 #[head,relation,tail,description,SEP,CLS,PAD]
        self.special_token_type_embeddings = nn.Embedding(max_special_token_type_ids, config.hidden_size)
        self.special_segment_embeddings    = nn.Embedding(100, config.hidden_size)
        
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(self, input_ids=None, special_token_type_ids=None, special_segment_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        special_token_type_embeddings = self.special_token_type_embeddings(special_token_type_ids)
        special_segment_embeddings    = self.special_segment_embeddings(special_segment_ids)

        embeddings = inputs_embeds
        if GlobalVariable.use_special_token_type_embeddings:
            #print("================use special token type embedding================")
            embeddings = embeddings + special_token_type_embeddings
        if GlobalVariable.use_special_segment_embeddings:
            #print("================use special segment embedding================")
            embeddings = embeddings + special_segment_embeddings
            
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings