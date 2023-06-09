import torch.nn as nn
from models.layers import FCBlock, FinalBlock
from models.vit import SimpleViT


class MLP(nn.Module):
    def __init__(self, opt):
        super(MLP, self).__init__()
        self.input = FCBlock(opt=opt, in_channels=28 * 28 * 3, out_channels=opt.width)
        self.hidden1 = FCBlock(opt=opt, in_channels=opt.width, out_channels=opt.width)
        self.dim_out = opt.width
        self.fc = FinalBlock(opt=opt, in_channels=opt.width)

    def forward(self, _x):
        _out = _x.view(_x.size(0), -1)
        _out = self.input(_out)
        _out = self.hidden1(_out)
        _out = self.fc(_out)
        return _out

class ViT_MLP(SimpleViT):
    def __init__(self,opt):
        super().__init__(
            image_size=28,
            patch_size=4,
            num_classes=opt.num_classes,
            dim=256,
            depth=3,
            heads=16,
            mlp_dim=256
        )