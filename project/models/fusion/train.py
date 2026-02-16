import sys
sys.path.append("/content/project")

import torch,torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import librosa,numpy as np
from transformers import BertTokenizer,BertModel
from utils.data_split import split
from utils.early_stop import EarlyStop

tok=BertTokenizer.from_pretrained("bert-base-uncased")
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

SR=16000; DUR=3; MAX=SR*DUR; MELS=128

class DS(Dataset):
    def __init__(self,s): self.s=s
    def __len__(self): return len(self.s)

    def __getitem__(self,i):
        x=self.s[i]
        a,_=librosa.load(x["audio"],sr=SR)
        a,_=librosa.effects.trim(a)
        a=np.pad(a,(0,max(0,MAX-len(a))))[:MAX]
        m=librosa.feature.melspectrogram(y=a,sr=SR,n_mels=MELS)
        m=librosa.power_to_db(m,ref=np.max)

        e=tok(x["text"],padding="max_length",
              truncation=True,max_length=16,
              return_tensors="pt")

        d={k:v.squeeze(0) for k,v in e.items()}
        d["mel"]=torch.tensor(m,dtype=torch.float32)
        d["label"]=torch.tensor(x["label"])
        return d

class Speech(nn.Module):
    def __init__(self):
        super().__init__()
        self.l=nn.LSTM(MELS,128,2,batch_first=True,bidirectional=True)
        self.f=nn.Linear(256,128)
    def forward(self,x):
        x=x.permute(0,2,1)
        o,_=self.l(x)
        return self.f(o.mean(1))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.s=Speech()
        self.t=BertModel.from_pretrained("bert-base-uncased")
        self.tp=nn.Linear(768,128)
        self.c=nn.Linear(256,7)

    def forward(self,m,i,a):
        s=self.s(m)
        t=self.t(i,attention_mask=a).last_hidden_state[:,0,:]
        t=self.tp(t)
        return self.c(torch.cat([s,t],1))

def run(m,l,c,o=None):
    t=o!=None
    m.train() if t else m.eval()
    tot=0
    with torch.set_grad_enabled(t):
        for b in l:
            mel=b["mel"].to(DEVICE)
            i=b["input_ids"].to(DEVICE)
            a=b["attention_mask"].to(DEVICE)
            y=b["label"].to(DEVICE)
            p=m(mel,i,a); loss=c(p,y)
            if t:
                o.zero_grad(); loss.backward(); o.step()
            tot+=loss.item()
    return tot/len(l)

def train():
    tr,va,_=split()
    tl=DataLoader(DS(tr),8,shuffle=True)
    vl=DataLoader(DS(va),8)

    m=Model().to(DEVICE)
    o=torch.optim.Adam(m.parameters(),1e-4)
    c=nn.CrossEntropyLoss()
    es=EarlyStop()

    for e in range(15):
        tloss=run(m,tl,c,o)
        vloss=run(m,vl,c)
        print(e+1,tloss,vloss)
        if es.step(vloss,m,"/content/project/models/fusion/best.pth"):
            break

if __name__=="__main__": train()
