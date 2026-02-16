import sys
sys.path.append("/content/project")

import torch,torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer,BertModel
from utils.data_split import split
from utils.early_stop import EarlyStop

tok=BertTokenizer.from_pretrained("bert-base-uncased")
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DS(Dataset):
    def __init__(self,s): self.s=s
    def __len__(self): return len(self.s)

    def __getitem__(self,i):
        x=self.s[i]
        e=tok(x["text"],padding="max_length",truncation=True,
              max_length=16,return_tensors="pt")
        d={k:v.squeeze(0) for k,v in e.items()}
        d["label"]=torch.tensor(x["label"])
        return d

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.b=BertModel.from_pretrained("bert-base-uncased")
        self.fc=nn.Linear(768,7)

    def forward(self,i,m):
        o=self.b(i,attention_mask=m).last_hidden_state[:,0,:]
        return self.fc(o)

def run(m,l,c,o=None):
    t=o!=None
    m.train() if t else m.eval()
    tot=0
    with torch.set_grad_enabled(t):
        for b in l:
            i=b["input_ids"].to(DEVICE)
            a=b["attention_mask"].to(DEVICE)
            y=b["label"].to(DEVICE)
            p=m(i,a); loss=c(p,y)
            if t:
                o.zero_grad(); loss.backward(); o.step()
            tot+=loss.item()
    return tot/len(l)

def train():
    tr,va,_=split()
    tl=DataLoader(DS(tr),8,shuffle=True)
    vl=DataLoader(DS(va),8)

    m=Model().to(DEVICE)
    o=torch.optim.AdamW(m.parameters(),2e-5)
    c=nn.CrossEntropyLoss()
    es=EarlyStop()

    for e in range(10):
        tloss=run(m,tl,c,o)
        vloss=run(m,vl,c)
        print(e+1,tloss,vloss)
        if es.step(vloss,m,"/content/project/models/text/best.pth"):
            break

if __name__=="__main__": train()
