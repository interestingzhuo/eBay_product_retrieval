
import os
import pdb

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.parallel.data_parallel import DataParallel
from skimage import measure
from pooling import *
import numpy as np
from normalization import *
from efficientnet_pytorch import EfficientNet

import timm
from models.wsdan import WSDAN
import queue
# import models as WSDAN

pool_dic = {
"GeM":GeM,
"SPoC":SPoC,
"MAC":MAC,
"RMAC":RMAC,
"GeMmp":GeMmp
 }






class ImageRetrievaleffNet(nn.Module):
    def __init__(self, net, pool):

        super(ImageRetrievaleffNet, self).__init__()
        self.net = net
        self.norm = L2N()
        self.pool = pool


    def forward(self, x, test=False):

        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = x.size(0)
        # Convolution layers
        x = self.net.extract_features(x)

        # Pooling and final linear layer
        # x = self.net._avg_pooling(x)
        x = self.pool(x)
        o = x
        x = x.view(bs, -1)
        x = self.net._dropout(x)
        x = self.net._fc(x)
        o = self.norm(o).squeeze(-1).squeeze(-1)

        return o, x

def normalize(A , symmetric=True):
    # A = A+I
    A = A + torch.eye(A.size(0)).cuda()
    # 所有节点的度
    d = A.sum(1)
    if symmetric:
        #D = D^-1/2
        D = torch.diag(torch.pow(d , -0.5))
        return D.mm(A).mm(D)
    else :
        # D=D^-1
        D =torch.diag(torch.pow(d,-1))
        return D.mm(A)
class ImageRetrievalresNet(nn.Module):
    def __init__(self, features,fc_cls,pool):
        
        super(ImageRetrievalresNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.pool = pool
        self.norm = L2N()
        if type(fc_cls)==list: 
          self.fc_cls = nn.Sequential(*fc_cls)   
        else:
          self.fc_cls=fc_cls
        
    def bfs(self, x: int, y: int, mask: torch.tensor, cc_map: torch.tensor, cc_id: int) -> int:
        dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        q = queue.LifoQueue()
        q.put((x, y))

        ret = 1
        cc_map[x][y] = cc_id

        while not q.empty():
            x, y = q.get()

            for (dx, dy) in dirs:
                new_x = x + dx
                new_y = y + dy
                if 0 <= new_x < mask.shape[0] and 0 <= new_y < mask.shape[1]:
                    if mask[new_x][new_y] == 1 and cc_map[new_x][new_y] == 0:
                        q.put((new_x, new_y))
                        ret += 1
                        cc_map[new_x][new_y] = cc_id
        return ret
    def find_max_cc(self, mask: torch.tensor) -> torch.tensor:
        """
        Find the largest connected component of the mask。
        Args:
            mask (torch.tensor): the original mask.
        Returns:
            mask (torch.tensor): the mask only containing the maximum connected component.
        """
        assert mask.ndim == 4
        assert mask.shape[1] == 1
        mask = mask[:, 0, :, :]
        for i in range(mask.shape[0]):
            m = mask[i]
            cc_map = torch.zeros(m.shape)
            cc_num = list()

            for x in range(m.shape[0]):
                for y in range(m.shape[1]):
                    if m[x][y] == 1 and cc_map[x][y] == 0:
                        cc_id = len(cc_num) + 1
                        cc_num.append(self.bfs(x, y, m, cc_map, cc_id))

            max_cc_id = cc_num.index(max(cc_num)) + 1
            m[cc_map != max_cc_id] = 0
        mask = mask[:, None, :, :]
        return mask
    def scda(self, o):
         
        # mask = o.sum(dim=1, keepdims=True)
        # thres = mask.mean(dim=(2, 3), keepdims=True)
        # mask[mask <= thres] = 0
        # mask[mask > thres] = 1
        # mask = self.find_max_cc(mask)
        # o = o * mask

        
        mask = o.squeeze()
        mask = torch.sum(mask,dim=1) 
        for i in range(o.shape[0]):
            
            mean = torch.mean(mask[i])
            mask[i] = mask[i] > mean
            mask[i] = mask[i].float()
  
            mask_t = mask[i].cpu().detach().numpy()

            testa1 = measure.label(mask_t, connectivity = 2)#8连通
            mask_t = np.zeros((mask_t.shape[0],mask_t.shape[1]))
            props = measure.regionprops(testa1)

            numPix = []
            for ia in range(len(props)):
                numPix += [props[ia].area]
            #像素最多的连通区域及其指引
            maxnum = max(numPix)
            index = numPix.index(maxnum)
            for j,k in props[index].coords:
               mask[i][j][k] = 1
        
        # pdb.set_trace()
        mask = mask.type(torch.FloatTensor).cuda()
        mask = mask.unsqueeze(dim = 1)
        o = o.mul(mask)
        
        return o
    def graph(self, o):
        
        # pdb.set_trace()
        o = o.permute(0,2,3,1)
        shape = o[0].shape
        for i in range(o.shape[0]):
            
            tmp = o[i].reshape((-1,2048))
            A = torch.mm(tmp, tmp.t())
            A = normalize(A,True)
            tmp = F.relu(self.g_fc1(A.mm(tmp)))
            tmp = F.relu(self.g_fc2(A.mm(tmp)))
            tmp = self.g_fc3(A.mm(tmp))
            # pdb.set_trace()
            
            tmp = tmp.transpose(0,1)
            o[i] = tmp.reshape(shape)
        return o

    def forward(self, x, test=False):
        o = self.features(x)
        
        if False:
            # o = self.graph(o)
            o = self.scda(o)
        # pdb.set_trace()
        
            
        
        o = self.pool(o)
        # pdb.set_trace()
        cls = self.fc_cls(o.squeeze())
        o = self.norm(o).squeeze(-1).squeeze(-1)
        return o, cls

class ImageRetrieval_WSDAN(nn.Module):
    def __init__(self, net):
        
        super(ImageRetrieval_WSDAN, self).__init__()
        self.net = net
        self.norm = L2N()
    
    def forward(self, x, test=False):
        cls, o, attention_map = self.net(x)
        o = self.norm(o).squeeze(-1).squeeze(-1)
        return o, cls
class SwinImageRetrievalNet(nn.Module):
    def __init__(self, net, pool):

        super(SwinImageRetrievalNet, self).__init__()
        self.net = net
        self.norm = L2N()
        self.pool = pool

    def forward(self, x, test=False):
        
        x = self.net.patch_embed(x)
        if self.net.absolute_pos_embed is not None:
            x = x + self.net.absolute_pos_embed
        x = self.net.pos_drop(x)
        x = self.net.layers(x)
        x = self.net.norm(x)  # B L C
        pdb.set_trace()
        x = x.transpose(1, 2)
        x = x.unsqueeze(-1)
        # x = self.net.avgpool(x.transpose(1, 2))  # B C 1
        o = x = self.pool(x)
        
        # o = x = self.net.forward_features(x)
        x = self.net.head(x)
        o = self.norm(o).squeeze(-1).squeeze(-1)
        return o, x
class VITImageRetrievalNet(nn.Module):
    def __init__(self, net):

        super(VITImageRetrievalNet, self).__init__()
        self.net = net
        self.norm = L2N()

    def forward(self, x, test=False):
        o = self.net.forward_features(x)
        x = self.net.head(o)
        o = self.norm(o).squeeze(-1).squeeze(-1)
        return o, x

def image_net(net_name,opt):
    if "R-" in opt.pool:
        if opt.pool == 'R-ori':
            pool = net.avgpool
        else:
            pool =  pool_dic[opt.pool[2:]]()
        pool = Rpool(pool)
    else:
        if opt.pool == 'ori':
            pool = net.avgpool
        else:
           pool = pool_dic[opt.pool]()

    if net_name == 'resnet101':
        net = torchvision.models.resnet101(pretrained=True)
        features = list(net.children())[:-2]
    elif net_name == 'resnet50':
        net = torchvision.models.resnet50(pretrained=True)
        features = list(net.children())[:-2]
        
    elif 'ibn' in net_name:
        net = model = torch.hub.load('XingangPan/IBN-Net', net_name, pretrained=True)
        features = list(net.children())[:-2]

    elif net_name == 'WSDAN':
        net = WSDAN(num_classes=opt.cls_num, M=32, net='inception_mixed_6e', pretrained=True)
        return ImageRetrieval_WSDAN(net)
        
    elif 'cspresnext50' in net_name :
        net = timm.create_model(net_name, pretrained = True)
        features = list(net.children())[:-1]
        # pdb.set_trace()
    elif 'legacy' in net_name :
        net = timm.create_model(net_name, pretrained = True)
        # pdb.set_trace()
        features = list(net.children())[:-2]
        
    elif 'vit' in net_name:
        net = timm.create_model(net_name, pretrained = True)
        net.head = nn.Linear(net.embed_dim, opt.cls_num)
        return VITImageRetrievalNet(net)
    elif 'swin' in net_name:
        net = timm.create_model(net_name, pretrained = True)
        net.head = nn.Linear(net.num_features, opt.cls_num)
        return SwinImageRetrievalNet(net,pool)
    elif 'ecaresnet' in net_name :
        net = timm.create_model(net_name, pretrained = True)
        # pdb.set_trace()
        features = list(net.children())[:-2]

    elif 'regnety_160' in net_name :
        net = timm.create_model(net_name, pretrained = True)
        features = list(net.children())[:-2]
        fc_cls = nn.Linear(in_features=1232, out_features=opt.cls_num, bias=True)
        return ImageRetrievalresNet(features,fc_cls,pool)
         
    elif 'efficient' in net_name:
        net = EfficientNet.from_pretrained(net_name, num_classes=opt.cls_num)
        return ImageRetrievaleffNet(net,pool)
    elif "resnest" in net_name:
        net = timm.create_model(net_name, pretrained = True)
        features = list(net.children())[:-2]
    else:
        
        raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))
    # pdb.set_trace()
    
    fc_cls = nn.Linear(in_features=2048, out_features=opt.cls_num, bias=True)
    return ImageRetrievalresNet(features,fc_cls,pool)


