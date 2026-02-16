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

    loader=DataLoader(DS(te),8)

    model=Model().to(DEVICE)
    model.load_state_dict(torch.load(
        "/content/project/models/fusion/best.pth",
        map_location=DEVICE
    ))

    model.eval()

    preds=[]
    labels=[]

    with torch.no_grad():
        for b in loader:
            mel=b["mel"].to(DEVICE)
            i=b["input_ids"].to(DEVICE)
            a=b["attention_mask"].to(DEVICE)

            out=model(mel,i,a)
            p=torch.argmax(out,1).cpu().numpy()

            preds.extend(p)
            labels.extend(b["label"].numpy())

    acc,f1=metrics(labels,preds)

    print("Fusion Accuracy:",acc)
    print("Fusion F1:",f1)

    confusion(labels,preds,
        "/content/project/results/plots/fusion_cm.png",
        "Fusion Confusion")

    return acc,f1

if __name__=="__main__":
    test()
