import sys
sys.path.append("/content/project")

import torch,torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import librosa,numpy as np
from utils.data_split import split
from utils.early_stop import EarlyStop

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
        return torch.tensor(m,dtype=torch.float32),torch.tensor(x["label"])

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(MELS,128,2,batch_first=True,bidirectional=True)
        self.fc=nn.Linear(256,7)

    def forward(self,x):
        x=x.permute(0,2,1)
        o,_=self.lstm(x)
        return self.fc(o.mean(1))

def run(m,l,c,o=None):
    t=o!=None
    m.train() if t else m.eval()
    tot=0
    with torch.set_grad_enabled(t):
        for x,y in l:
            x,y=x.to(DEVICE),y.to(DEVICE)
            p=m(x); loss=c(p,y)
            if t:
                o.zero_grad(); loss.backward(); o.step()
            tot+=loss.item()
    return tot/len(l)

def train():
    tr,va,_=split()
    tl=DataLoader(DS(tr),16,shuffle=True)
    vl=DataLoader(DS(va),16)

    m=Model().to(DEVICE)
    o=torch.optim.Adam(m.parameters(),1e-4)
    c=nn.CrossEntropyLoss()
    es=EarlyStop()

    for e in range(20):
        tloss=run(m,tl,c,o)
        vloss=run(m,vl,c)
        print(e+1,tloss,vloss)
        if es.step(vloss,m,"/content/project/models/speech/best.pth"):
            break

if __name__=="__main__": train()
