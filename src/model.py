
import torch.nn as nn
import torch
from transformers import RobertaModel
import os

class ToxicityClassifier(nn.Module):
    def __init__(self):
        super(ToxicityClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.roberta.config.hidden_size, 2) 

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # get the CLS token
        pooled_output = outputs.pooler_output 
        logits = self.fc(self.dropout(pooled_output))
        return logits

def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")