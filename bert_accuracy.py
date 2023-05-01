from transformers import RobertaModel, RobertaTokenizer
import torch.nn  as nn
from torch import load

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
tokenizer.add_tokens(["<u>"])



class CustomRobertaModel(nn.Module):
  def __init__(self, dropout, out_classes):
    super().__init__()
    self.RBERT = RobertaModel.from_pretrained("roberta-base")
    self.RBERT.resize_token_embeddings(len(tokenizer))

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
  

model = CustomRobertaModel(
    dropout=0.4,
    out_classes=1
)


data = load("acc-part-1.pt")
model.load_state_dict(data)


def accuracy(string1, string2):
  sen = string1+"<u"+string2
  sen = tokenizer(sen, return_tensors='pt')
  sen = model(**sen)

  return round(sen.item(), 2)
