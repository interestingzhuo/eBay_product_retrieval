
import torchvision 
import pdb
import os
import torchvision.transforms as transforms
from dataset import *
import torch
import torch.nn as nn
from utils import *
import time
import numpy as np
from imageretrievalnet import *
from loss import *
import time
import argparse
from MAP import *
from test import test_single_dataset
import batchminer    as bmine
import criteria      as criteria
import parameters    as par
from torch.cuda.amp import *
from autoaugment import CIFAR10Policy
# from RA_CNN import *
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def gimme_save_string(opt):
    varx = vars(opt)
    base_str = ''
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key],dict):
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t'+str(sub_key)+': '+str(sub_item)
        else:
            base_str += '\n\t'+str(varx[key])
        base_str+='\n\n'
    return base_str


parser = argparse.ArgumentParser()
parser = par.basic_training_parameters(parser)
parser = par.batch_creation_parameters(parser)
parser = par.batchmining_specific_parameters(parser)
parser = par.loss_specific_parameters(parser)
parser = par.wandb_parameters(parser)
parser = par.origin_parameters(parser)
opt = parser.parse_args()

def main():


    log_str = gimme_save_string(opt)

    EPOCHS = opt.epoch

    BATCH_SIZE = opt.bs
    
    image_size = opt.imsize
    
    lr = opt.lr

    dataset = opt.dataset
 
    root = opt.dataroot
    
    ann_folder = os.path.join(root, 'retrieval_dict')
    
    train_file = os.path.join(ann_folder, 'train.txt')
    
    val_file = os.path.join(ann_folder, 'val.txt')
    
    imgs_root = os.path.join(root)

    net_name = opt.net

    cls_num = opt.cls_num

    opt.n_classes = opt.cls_num

    ###############################################
    batch_p = opt.batch_p

    batch_k = opt.batch_k
    
    opt.bs = batch_p*batch_k
    
    


    model = image_net(net_name,opt).cuda()
    # model = RA_CNN(opt).cuda()
    if opt.resume != None:
        checkpoint = torch.load(opt.resume)

        if isinstance(checkpoint,DataParallel):
            checkpoint = checkpoint.module.state_dict()
        
        model.load_state_dict(checkpoint)
        del checkpoint
        torch.cuda.empty_cache()


 

    ####################################################
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = CIFAR10Policy()
   
    


    opt.device = torch.device('cuda')

    localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    d = localtime + '_' + opt.loss +  '_' +  opt.net
    
    if opt.graph:
        d+='_graph'

    directory = os.path.join(dataset,d)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory,"Parametre.txt"),'w') as f:
        f.write(log_str)    
    if opt.loss == 'cross':
        train_dataset = ImagesForCls_list(imgs_root, train_file, image_size,transform=transform)
        train_dataset_cls = train_dataset
        BATCH_SIZE_CLS = BATCH_SIZE
    else:
        train_dataset_cls = ImagesForCls_list(imgs_root, train_file, image_size,transform=transform)
        train_dataset = TuplesDataset_list(imgs_root, train_file, image_size,batch_p = batch_p,batch_k = batch_k,transform=transform)
        
        BATCH_SIZE_CLS = batch_p * batch_k


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=opt.kernels, pin_memory=True, sampler=None,
    )
    
    train_loader_cls = torch.utils.data.DataLoader(
        train_dataset_cls, batch_size=BATCH_SIZE_CLS, shuffle=True,
        num_workers=opt.kernels, pin_memory=True, sampler=None,
    )
    
    print('train dataloader finished!')
    

    test_dataset = ImagesForCls_list(imgs_root, val_file, image_size,transform=transform,is_validation=True)
    BATCH_SIZE = 512
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=8, pin_memory=True, sampler=None,)
    
    
    
    to_optim   = [{'params':model.parameters(),'lr':lr,'momentum':opt.momentum,'weight_decay':1e-5}]
    
    
    # criterion_cls =  nn.CrossEntropyLoss()
    criterion_cls, to_optim = criteria.select('arcface', opt, to_optim)
    criterion_cls.cuda()

    #################### LOSS SETUP ####################
    if opt.loss != 'cross':
        batchminer   = bmine.select(opt.batch_mining, opt)
    
        criterion_metric, to_optim = criteria.select(opt.loss, opt, to_optim, batchminer = batchminer)
        criterion_metric.cuda()

        
    
    

    if opt.optim == 'adam':
        optimizer    = torch.optim.Adam(to_optim)
    elif opt.optim == 'sgd':
        optimizer    = torch.optim.SGD(to_optim, momentum=0.9)
    else:
        raise Exception('Optimizer <{}> not available!'.format(opt.optim))

    # exp_decay = math.exp(-0.01)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)
    scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.tau, gamma=opt.gamma)


    if opt.mg:
        model=nn.DataParallel(model,device_ids=[0,1,2,3,4,5,6,7]) 
    
    Logger_file = os.path.join(directory,"log.txt")
    
    if opt.test:
        
        # test_single_dataset(model)
        # metric = ''
        AP,precision = test(test_loader, model, -1)
        precision = 'precision: ' + str(precision)
        metric = precision
        AP = '\tAverage Precisioni: ' + str(AP)
        metric += AP
        print(metric)
        with open(Logger_file,'a') as f:
            f.write(metric+'\n')
    tmp = opt.lamda
    for epoch in range(EPOCHS):
        

        
        if opt.loss != 'cross':
            

            if epoch % 20 == 0 and epoch != 0:#???????????????
                opt.lamda = 1
                train(train_loader_cls,model,epoch,criterion_cls,optimizer,opt)
            else:
                opt.lamda = tmp
                train_loader.dataset.create_tuple()
                print('tuple created finished!')
                train(train_loader,model,epoch,criterion_cls,optimizer,opt,criterion_metric)

            
        else:
            train(train_loader,model,epoch,criterion_cls,optimizer,opt)
        
        scheduler.step()
        torch.cuda.empty_cache()
        
        # metric = test_single_dataset(model)
        metric = ''
        AP,precision = test(test_loader, model, -1)
        precision = 'precision: ' + str(precision)
        metric += precision
        AP = '\t Average Precision: ' + str(AP)
        metric += AP 
        with open(Logger_file,'a') as f:
            f.write(metric+'\n')
            #f.write("epoch:{}\tAP@m:{}\tPrecision:{}\tmAP:{}\trecall:{}\n".format(epoch,AP,precision,mAP,recall))
        path = os.path.join(directory,'model_epoch_{}.pth'.format(epoch))
        if isinstance(model,DataParallel):
            torch.save(model.module.state_dict(),path)
        else:
            torch.save(model.state_dict(),path)
        
        #if epoch%2 == 1:
        #    for i in range(len(optimizer.param_groups)):
        #        optimizer.param_groups[i]['lr'] /= 2
             

def train(train_loader,model,epoch,criterion_cls, optimizer,opt, criterion_metric=None):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    m_loss = AverageMeter() 
    cls_loss = AverageMeter()
    end = time.time()
    model.train()
    # scaler = GradScaler()
    for step, (x, cls) in enumerate(train_loader):
        batch_time.update(time.time() - end)
        end = time.time()
        x = x.squeeze()
        cls = cls.squeeze()
        
        x = x.cuda()
        # with autocast():
        out,output = model(x)
    
    
        # loss = (1-opt.lamda)*criterion_cls(output ,cls.cuda())#????????????
        loss = (1-opt.lamda)*criterion_cls(out ,cls.cuda())
        # pdb.set_trace()
        cls_loss.update(loss.item())
        if criterion_metric != None:
            metric_loss = opt.lamda*criterion_metric(out,cls)#????????????
            m_loss.update(metric_loss.item())
            loss = loss + metric_loss

        train_loss.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=20, norm_type=2)
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        lr = optimizer.param_groups[0]['lr']
        if step % 100 == 0:
            print('>> Train: [{0}][{1}/{2}]\tlr: {3}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {train_loss.val:.3f} ({train_loss.avg:.3f})\t'
                'Metric loss {m_loss.val:.3f} ({m_loss.avg:.3f})\t'
                'Class loss {cls_loss.val:.3f} ({cls_loss.avg:.3f})\t'
                .format(
                epoch+1, step+1, len(train_loader),lr, batch_time=batch_time,
                train_loss=train_loss,m_loss=m_loss,cls_loss=cls_loss))

def test(test_loader, model, epoch):
    print('>> Evaluating network on test datasets...')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.eval()
    ap_meter = AveragePrecisionMeter(False)
    precision = PrecisionMeter(False)
    right = 0
    cnt = 0
    dataset = []
    gt = []
    for step, (x, lbl) in enumerate(test_loader):
        batch_time.update(time.time() - end)
        end = time.time()
        x = x.cuda()
        x = x.contiguous()

        with torch.no_grad():
            embd, output = model(x)
        dataset.extend(embd.unsqueeze(0))
        gt.extend(lbl)
        
        target = lbl.cuda()
        output = output.argmax(dim = 1)
        precision =  target == output
        right += sum(precision)
        cnt += len(precision)

        precision = float(right)/cnt
        # precision = 0
        if step % 100 == 0:
            print('>> Test: [{0}][{1}/{2}]\t precision: {3}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                epoch+1, step+1, len(test_loader), precision,  batch_time=batch_time,
                data_time=data_time))
    dataset = torch.cat(dataset,dim=0)
    gt = np.reshape(gt,-1)
    AP, recall = Test_mAP(dataset,gt)
    return AP, precision
if __name__=='__main__':
    main()
