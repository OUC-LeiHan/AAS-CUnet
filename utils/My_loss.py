import torch 
import torch.nn as nn

class balanced_Mse(nn.Module):
    def __init__(self,b=0.5):
        super(balanced_Mse,self).__init__()
        self.b = b

    def forward(self,x,y,geo):
        print(geo.size())
        print(x.size())
        Mse = torch.mean(torch.pow((x-y),2)*geo)
        Mae = torch.mean(torch.abs((x-y))*geo)
        return self.b*Mse+(1-self.b)*Mae


