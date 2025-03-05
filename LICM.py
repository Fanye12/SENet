#This part of the code refers to convpass https://github.com/JieShibo/PETL-ViT/blob/main/convpass/vtab/convpass.py
import torch
from torch import nn
import timm
import torch.nn.functional as F

def forward_block(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s  #
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x))) * self.s
    return x

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
      
class LICM(nn.Module):
    def __init__(self, dim=768, xavier_init=True):
        super().__init__()

        self.adapter_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.adapter_up = nn.Linear(dim, 768)    # equivalent to 1 * 1 Conv
        self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
        self.act = QuickGELU()
   
    def forward(self, x):
        B, N, C = x.shape
        x = self.act(self.adapter_up(x))
        x = x.reshape(B*N, 16, 16, 3).permute(0, 3, 1, 2)
        x = self.adapter_conv(x)
        x = x.permute(0, 2, 3, 1).reshape(B, N, 768)
        x = self.adapter_down(self.act(x))
        return x
    
def set_LICM(model, s=1):
    for _ in model.children():
        if type(_) == timm.models.vision_transformer.Block:
            if _.norm1.normalized_shape[0] == 512:
                _.adapter_attn = LICM(dim=512)
                _.adapter_mlp = LICM(dim=512)
                _.s = s
                bound_method = forward_block.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif _.norm1.normalized_shape[0] == 768:
                _.adapter_attn = LICM(dim=768)
                _.adapter_mlp = LICM(dim=768)
                _.s = s
                bound_method = forward_block.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            
        elif len(list(_.children())) != 0:
            set_LICM(_, s=s)

   
