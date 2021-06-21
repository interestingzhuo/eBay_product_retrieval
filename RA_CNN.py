import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from imageretrievalnet import *

class RACNN(nn.Module):
    def __init__(self, loc_net, embed_net):
        super(RACNN, self).__init__()

        self.loc_net = loc_net
        
        self.classifier = nn.Linear(2048, 1000)
        self.location_layer = nn.Linear(2048, 4)
        self.pool = 
        self.crop_resize = AttentionCropLayer()

    def forward(self, x):
        feature = self.loc_net(x)
        atten =self.location_layer
        scaledA_x = self.crop_resize(x, atten * 224)
       
        emd, logit = self.embed_net(scaledA_x)
        return emd, logit

# class AttentionCropFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(self, images, locs):
#         h = lambda x: 1. / (1. + torch.exp(-10. * x))
#         in_size = images.size()[2]
#         unit = torch.stack([torch.arange(0, in_size)] * in_size).float()
#         x = torch.stack([unit.t()] * 3)
#         y = torch.stack([unit] * 3)
#         if isinstance(images, torch.cuda.FloatTensor):
#             x, y = x.cuda(), y.cuda()
        
#         in_size = images.size()[2]
#         ret = []
#         for i in range(images.size(0)):
#             w_off,w_end, h_off, h_end = locs[i][0], locs[i][1], locs[i][2], locs[i][3]
#             mk = (h(x-w_off) - h(x-w_end)) * (h(y-h_off) - h(y-h_end))
#             xatt = images[i] * mk  
#             ret.append(xatt)
        
#         ret_tensor = torch.stack(ret)
#         self.save_for_backward(images, ret_tensor)
#         return ret_tensor

#     @staticmethod
#     def backward(self, grad_output):
#         images, ret_tensor = self.saved_variables[0], self.saved_variables[1]
#         in_size = 224
#         ret = torch.Tensor(grad_output.size(0), 3).zero_()
#         norm = -(grad_output * grad_output).sum(dim=1)
        
        
#         x = torch.stack([torch.arange(0, in_size)] * in_size).t()
#         y = x.t()
#         long_size = (in_size/3*2)
#         short_size = (in_size/3)
#         mx = (x >= long_size).float() - (x < short_size).float()
#         my = (y >= long_size).float() - (y < short_size).float()
#         ml = (((x<short_size)+(x>=long_size)+(y<short_size)+(y>=long_size)) > 0).float()*2 - 1
        
#         mx_batch = torch.stack([mx.float()] * grad_output.size(0))
#         my_batch = torch.stack([my.float()] * grad_output.size(0))
#         ml_batch = torch.stack([ml.float()] * grad_output.size(0))
        
#         if isinstance(grad_output, torch.cuda.FloatTensor):
#             mx_batch = mx_batch.cuda()
#             my_batch = my_batch.cuda()
#             ml_batch = ml_batch.cuda()
#             ret = ret.cuda()
        
#         ret[:, 0] = (norm * mx_batch).sum(dim=1).sum(dim=1)
#         ret[:, 1] = (norm * my_batch).sum(dim=1).sum(dim=1)
#         ret[:, 2] = (norm * ml_batch).sum(dim=1).sum(dim=1)
#         return None, ret

class AttentionCropLayer(nn.Module):
    """
        Crop function sholud be implemented with the nn.Function.
        Detailed description is in 'Attention localization and amplification' part.
        Forward function will not changed. backward function will not opearate with autograd, but munually implemented function
    """
    def forward(self, images, locs):

        h = lambda x: 1. / (1. + torch.exp(-10. * x))
        in_size = images.size()[2]
        unit = torch.stack([torch.arange(0, in_size)] * in_size).float()
        x = torch.stack([unit.t()] * 3)
        y = torch.stack([unit] * 3)
        if isinstance(images, torch.cuda.FloatTensor):
            x, y = x.cuda(), y.cuda()
        
        in_size = images.size()[2]
        ret = []
        for i in range(images.size(0)):
            w_off,w_end, h_off, h_end = locs[i][0], locs[i][1], locs[i][2], locs[i][3]
            mk = (h(x-w_off) - h(x-w_end)) * (h(y-h_off) - h(y-h_end))
            xatt = images[i] * mk  
            ret.append(xatt)
        
        ret_tensor = torch.stack(ret)
        # self.save_for_backward(images, ret_tensor)
        return ret_tensor
        # return AttentionCropFunction.apply(images, locs)

def RA_CNN(opt):
    net_name = opt.net
    opt.cls_num = 1024
    loc_net = image_net(net_name,opt)
    opt.cls_num = 1000
    embed_net = image_net(net_name,opt)
    return RACNN(loc_net, embed_net);
   