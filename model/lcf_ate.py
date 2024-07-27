
# from pytorch_transformers.modeling_bert import BertForTokenClassification, BertPooler, BertSelfAttention
from transformers.models.bert.modeling_bert import BertForTokenClassification, BertPooler, BertSelfAttention

from torch.nn import Linear, CrossEntropyLoss
import torch
import torch.nn as nn
import copy
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_vec = np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_length))
        zero_tensor = torch.tensor(zero_vec).float().to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class LCF_ATEPC(BertForTokenClassification):

    def __init__(self, bert_base_model, args):
        super(LCF_ATEPC, self).__init__(config=bert_base_model.config)
        config = bert_base_model.config
        self.bert_for_global_context = bert_base_model
        self.args = args
        # do not init lcf layer if BERT-SPC or BERT-BASE specified
        # if self.args.local_context_focus in {'cdw', 'cdm', 'fusion'}:
        if not self.args.use_unique_bert:
            self.bert_for_local_context = copy.deepcopy(self.bert_for_global_context)
        else:
            self.bert_for_local_context = self.bert_for_global_context
        self.pooler = BertPooler(config)
        self.bert_global_focus = self.bert_for_global_context
        self.dropout = nn.Dropout(self.args.dropout)
        self.SA1 = SelfAttention(config, args)
        self.SA2 = SelfAttention(config, args)
        self.linear_double = nn.Linear(768 * 2, 768)
        self.linear_triple = nn.Linear(768 * 3, 768)

    def get_batch_token_labels_bert_base_indices(self, labels):
        if labels is None:
            return
        # convert tags of BERT-SPC input to BERT-BASE format
        labels = labels.detach().cpu().numpy()
        for text_i in range(len(labels)):
            sep_index = np.argmax((labels[text_i] == 5))
            labels[text_i][sep_index + 1:] = 0
        return torch.tensor(labels).to(self.args.device)

    def get_ids_for_local_context_extractor(self, text_indices):
        # convert BERT-SPC input to BERT-BASE format
        text_ids = text_indices.detach().cpu().numpy()
        for text_i in range(len(text_ids)):
            sep_index = np.argmax((text_ids[text_i] == 102))
            text_ids[text_i][sep_index + 1:] = 0
        return torch.tensor(text_ids).to(self.args.device)

    def forward(self, input_ids_spc, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None):
        if not self.args.use_bert_spc:
            input_ids_spc = self.get_ids_for_local_context_extractor(input_ids_spc)
            labels = self.get_batch_token_labels_bert_base_indices(labels)
        global_context_out = self.bert_for_global_context(input_ids_spc, token_type_ids, attention_mask)[
            'last_hidden_state']

        batch_size, max_len, feat_dim = global_context_out.shape
        global_valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).to(self.args.device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    global_valid_output[i][jj] = global_context_out[i][j]
        global_context_out = self.dropout(global_valid_output)
        ate_logits = self.classifier(global_context_out)

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            loss_ate = loss_fct(ate_logits.view(-1, self.num_labels), labels.view(-1))

            return loss_ate
        else:
            return ate_logits
