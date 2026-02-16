import os
from sklearn.model_selection import train_test_split

BASE = "/content/project/data/TESS Toronto emotional speech set data"

EMOTIONS = ["angry","disgust","fear","happy","neutral","sad","surprise"]
LABEL = {e:i for i,e in enumerate(EMOTIONS)}

def collect():
    samples=[]
    for folder in os.listdir(BASE):
        emotion="_".join(folder.split("_")[1:]).lower()
        if emotion in ["pleasant_surprise","pleasant_surprised"]:
            emotion="surprise"

        fp=os.path.join(BASE,folder)
        for f in os.listdir(fp):
            word=f.split("_")[2].replace(".wav","")
            samples.append({
                "audio":os.path.join(fp,f),
                "text":f"Say the word {word}",
                "label":LABEL[emotion]
            })
    return samples

def split(seed=42):
    s=collect()
    y=[x["label"] for x in s]

    trv,te=train_test_split(s,test_size=0.2,stratify=y,random_state=seed)
    ytv=[x["label"] for x in trv]
    tr,va=train_test_split(trv,test_size=0.1,stratify=ytv,random_state=seed)

    return tr,va,te
