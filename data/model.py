import torch.nn as nn
import transformers


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.model = transformers.BertModel.from_pretrained("bert-base-uncased", return_dict=False)
        self.model_drop = nn.Dropout(0.3)
        self.linear = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        out_1, out_2 = self.model(ids,
                                  attention_mask=mask,
                                  token_type_ids=token_type_ids
                                  )
        return self.linear(self.model_drop(out_2))
