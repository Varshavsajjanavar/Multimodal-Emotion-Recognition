import torch

class EarlyStop:
    def __init__(self,patience=3):
        self.best=1e9
        self.p=patience
        self.c=0

    def step(self,loss,model,path):
        if loss<self.best:
            self.best=loss
            self.c=0
            torch.save(model.state_dict(),path)
            return False
        self.c+=1
        return self.c>=self.p
