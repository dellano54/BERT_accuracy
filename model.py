import torch.nn  as nn

class CustomRobertaModel(nn.Module):
  def __init__(self, RBERT, dropout, out_classes):
    super().__init__()
    self.RBERT = RBERT
    self.drop = nn.Dropout(dropout)
    self.Linear = nn.Sequential(
        nn.Linear(self.RBERT.config.hidden_size, out_classes)
        )

    self.act = nn.Sigmoid()

  def forward(self, input_ids, attention_mask):
    pred = self.RBERT(input_ids=input_ids,
                      attention_mask=attention_mask).pooler_output

    pred = self.drop(pred)
    pred = self.Linear(pred)
    pred = self.act(pred)

    return pred
