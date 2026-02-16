import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

EMOTIONS = ["angry","disgust","fear","happy","neutral","sad","surprise"]

def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return acc, f1

def confusion(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8,6))
    plt.imshow(cm)
    plt.title(title)

    plt.xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45)
    plt.yticks(range(len(EMOTIONS)), EMOTIONS)

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i,j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
