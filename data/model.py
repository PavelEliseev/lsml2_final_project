import torch.nn as nn
import transformers


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased", return_dict=False)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        out_1, out_2 = self.bert(ids,
                                  attention_mask=mask,
                                  token_type_ids=token_type_ids
                                  )
        return self.out(self.bert_drop(out_2))
