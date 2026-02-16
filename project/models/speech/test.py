import sys
sys.path.append("/content/project")

import torch
from torch.utils.data import DataLoader
from train import DS, Model
from utils.data_split import split
from utils.eval import metrics, confusion

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test():
    _,_,te=split()

    loader=DataLoader(DS(te),16)

    model=Model().to(DEVICE)
    model.load_state_dict(torch.load(
        "/content/project/models/speech/best.pth",
        map_location=DEVICE
    ))

    model.eval()

    preds=[]
    labels=[]

    with torch.no_grad():
        for x,y in loader:
            x=x.to(DEVICE)
            out=model(x)
            p=torch.argmax(out,1).cpu().numpy()

            preds.extend(p)
            labels.extend(y.numpy())

    acc,f1=metrics(labels,preds)

    print("Speech Accuracy:",acc)
    print("Speech F1:",f1)

    confusion(labels,preds,
        "/content/project/results/plots/speech_cm.png",
        "Speech Confusion")

    return acc,f1

if __name__=="__main__":
    test()
