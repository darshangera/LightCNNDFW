#Aum Sri Sai Ram Ganesayah Namah

#Training on DFW dataset train pairs(400 subjects) and validation on validation set(600 subjects)
# use files :dfw_traindata_pairs_only123.txt(train) and dfw_testdata_pairs_only123.txt(validation)
#Protocol1: Impersonation +ve:1 and -ve :3
#Used Lightcnn 29 model for fine tuning with Siamese network with Binary Cross Entropy loss


from __future__ import print_function
import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import OrderedDict

import numpy as np
import cv2

from Contrastive import ContrastiveLoss
from load_imglist import ImageList
from eval_metrics import evaluate

from light_cnn import LightCNN_4Layers, LightCNN_9Layers, LightCNN_29Layers_v2#LightCNN_29_v2Layers



parser = argparse.ArgumentParser(description='PyTorch DFW Training')

parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')

parser.add_argument('--cuda', '-c', default=True)

parser.add_argument('--batch_size', type=int, default = 32 , metavar='N',
                        help='input batch size for training (default: 64)')

parser.add_argument('--epochs', type=int, default = 80, metavar='N',
                        help='number of epochs to train (default: 10)')


parser.add_argument('--start-epoch', default = 0, type=int, metavar='N',
                help='manual epoch number (useful on restarts)')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                    help='SGD momentum (default: 0.5)')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--pretrained', default = True, type = bool,
                metavar='N', help='use pretrained ligthcnn model:True / False no pretrainedmodel )')

parser.add_argument('--resume', default='./LightCNN_29_V2Layers_checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--model', default='LightCNN-29v2', type=str, metavar='Model',
                    help='model type: LightCNN-9, LightCNN-29')

parser.add_argument('--root_path', default='../DisguisedFacesInTheWild/', type=str, metavar='PATH',
                    help='root path of face images (default: none).')

parser.add_argument('--train_list', default='../DisguisedFacesInTheWild/dfw_traindata_pairs_only123.txt', type=str, metavar='PATH',
                    help='list of face images for feature extraction (default: none).')

parser.add_argument('--val_list', default='../DisguisedFacesInTheWild/dfw_testdata_pairs_only123.txt', type=str, metavar='PATH',
                    help='list of face images for feature extraction (default: none).')

parser.add_argument('--save_path', default='./', type=str, metavar='PATH',
                    help='save root path for features of face images.')
parser.add_argument('--num_classes', default=80013, type=int,
                    metavar='N', help='mini-batch size (default: 79077)')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')

parser.add_argument('--print-freq', '-p', default=100, type=int,
                   metavar='N', help='print frequency (default: 100)')


class DFW(nn.Module):  #Model for finetuning with BCE loss
        def __init__(self):
            super(DFW, self).__init__()
            self.main = nn.Sequential(
             nn.Linear(512,1),
             nn.Sigmoid()
            )
        def forward(self,input):
            output =self.main(input)
            return output



def main():
    global args
    args = parser.parse_args()

    if args.model == 'LightCNN-9':
        model = LightCNN_9Layers(num_classes=args.num_classes)
    elif args.model == 'LightCNN-29':
        model = LightCNN_29Layers(num_classes=args.num_classes)
    elif args.model == 'LightCNN-29v2':
        model = LightCNN_29Layers_v2(num_classes=args.num_classes)
    else:
        print('Error model type\n')

    use_cuda = args.cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    print('Device being used is :' + str(device))

    #model = torch.nn.DataParallel(model).to(device)
    model = model.to(device)
    DFWmodel = DFW().to(device)


    if args.pretrained:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            #checkpoint = torch.load(args.resume, map_location='cpu')['state_dict']
            if device == 'cpu':
                state_dict = torch.load(args.resume, map_location='cpu')['state_dict']#torch.load(directory, map_location=lambda storage, loc: storage)
            else:
                state_dict =  torch.load(args.resume, map_location = lambda storage, loc: storage)['state_dict']

            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                if k[:7] == 'module.':
                    name = k[7:] # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict,strict=True)
            #model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


    #load image
    train_loader = torch.utils.data.DataLoader(
        ImageList(root=args.root_path, fileList=args.train_list,
            transform=transforms.Compose([
                transforms.Resize((128,128)),
                #transforms.Resize((144,144)),
                #transforms.FiveCrop((128,128)),
                #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])),
        batch_size=args.batch_size, shuffle=False,       num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        ImageList(root=args.root_path, fileList=args.val_list,
            transform=transforms.Compose([
                transforms.Resize((128,128)),
                transforms.ToTensor(),
            ])),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    '''
    for param in list(model.named_parameters()):
        print(param)
    '''
    for name,param in model.named_parameters():
        if 'fc' in name and 'fc2' not in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    '''
    for name,param in model.named_parameters():
        print(name, param.requires_grad)
    '''

    params =  list(model.fc.parameters())+ list(DFWmodel.parameters())  #learnable parameters are fc layer of lightcnn and DFWModel parameters

    optimizer = optim.SGD(params , lr=args.lr, momentum=args.momentum)
    #optimizer = optim.Adam(params , lr=args.lr)

    #criterion   = ContrastiveLoss(margin = 1.0 ).to(device)
    criterion   = nn.BCELoss()#ContrastiveLoss(margin = 1.0 ).to(device)

    for epoch in range(args.start_epoch, args.epochs):

        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, DFWmodel, criterion, optimizer, epoch, device)

        # evaluate on validation set
        acc = validate(val_loader, model,DFWmodel, criterion,epoch, device)
        if epoch%10==0:
            save_name = args.save_path + 'lightCNN_' + str(epoch+1) + '_checkpoint.pth.tar'
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'acc': acc,
                'optimizer' : optimizer.state_dict(),
                    }, save_name)

def train(train_loader, model, DFWmodel,  criterion, optimizer, epoch, device):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    acc       = AverageMeter()


    model.train()
    labels, distance , distances = [], [], []

    end = time.time()
    accuracy = 0.0
    for i, (img1,img2, label) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img1      = img1.to(device)
        img2      = img2.to(device)
        '''
        bs,ncrops,c,h,w = img1.size()
        img1 = img1.view(-1,c,h,w)
        img2 = img2.view(-1,c,h,w)
        '''
        #label = torch.tensor(label,dtype=torch.int64)
        label = label.to(device)
        # compute output
        _ , feature1 = model(img1)
        _ ,  feature2 = model(img2)
        '''
        feature1 = feature1.view(bs,ncrops,-1).mean(1)
        feature2 = feature2.view(bs,ncrops,-1).mean(1)
        '''
        #print(feature1.size())
        feature = torch.cat((feature1,feature2),dim=1)
        output = DFWmodel(feature)
        #print(output.size(),feature.size())
        loss   = criterion(output, label)
        losses.update(loss.item(), img1.size(0))
        label = label.type(torch.LongTensor).to(device)

        t = torch.FloatTensor([0.5])  # threshold
        out = (output > t.cuda(async=True)).float() * 1
        #print(out[0:10],label[0:10])

        equals = label.float()  ==  out#.t()
        #print(equals[0:10])
        #print(torch.sum(equals))
        accuracy = (torch.sum(equals).cpu().numpy())
        # print(equals)


        #prec1 = accuracy(output.data, label, topk=(1,))
        # measure accuracy and record loss
        acc.update(accuracy,img1.size(0))
        # measure accuracy and record loss
        #dists  =  F.cosine_similarity(feature1,feature2)   #For Accuracy calculation by Facenet code
        #prec,_= cal_accuracy(dists.detach().cpu(),label.cpu())

        #print('dist',dists.size())
        #distance.append(dists.detach().cpu().numpy())
        #labels.append(label.data.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.fc.parameters(), 10)
        optimizer.step()
        #acc.update(prec, img1.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    #labels = np.array([l for label in labels  for l in label ])

    #distances  =  np.array([d for dist in distance for d in dist ])



    #print(labels.shape,distances.shape, distances[0:10], labels[0:10])

    #accuracy = evaluate(1-distances,labels)
    #accuracy1 = evaluate(distances,labels)
    #accuracy2,_ = cal_accuracy(1-distances,labels)

    #print(accuracy)
    #print(accuracy1)
    #print(accuracy2)
    #print(accuracy2)
    #print('Train set: Accuracy: {:.8f}\n\33[0m'.format(np.mean(accuracy)))
    print(acc.avg)
    return acc.avg


def validate(valid_loader, model, DFWmodel, criterion,  epoch, device):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    acc       = AverageMeter()



    model.eval()
    labels, distance , distances = [], [], []
    accuracy = 0.0
    end = time.time()
    with torch.no_grad():
       for i, (img1,img2, label) in enumerate(valid_loader):
           data_time.update(time.time() - end)

           img1      = img1.to(device)
           img2      = img2.to(device)
           label = label.to(device)
           _ , feature1 = model(img1)
           _ ,  feature2 = model(img2)
           feature = torch.cat((feature1,feature2),dim=1)
           output = DFWmodel(feature)
           loss   = criterion(output, label)

           label = label.type(torch.LongTensor).to(device)
           #prec1 = accuracy(output.data, label, topk=(1,))

           # measure accuracy and record loss
           losses.update(loss.item(), img1.size(0))

           t = torch.FloatTensor([0.5])  # threshold
           out = (output > t.cuda(async=True)).float() * 1
           #print(out.size(),label.size())

           equals = label.float()  ==  out
           #print(equals)
           #print(torch.sum(equals))
           accuracy = (torch.sum(equals).cpu().numpy())
           # print(equals)


           #prec1 = accuracy(output.data, label, topk=(1,))
           # measure accuracy and record loss
           acc.update(accuracy,img1.size(0))



           #acc.update(prec1[0],img1.size(0))

           batch_time.update(time.time() - end)
           end = time.time()

           if i % args.print_freq == 0:
               print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))


    print(acc.avg)
    return acc.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count
        #print(self.val,self.sum,self.count,self.avg)




def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    print(output)
    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    print(pred,target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
'''
def accuracy(feature1,feature2,label):
    cos_sim = []
    labels = []
    for i in range(len(label)):

        sim = cosin_metric(feature1[i],feature2[i])
        cos_sim.append(sim)
        labels.append(label[i])
    acc = cal_accuracy(np.array(cos_sim),np.array(labels))
    return acc
'''
def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def save_checkpoint(state, filename):
    torch.save(state, filename)

if __name__ == '__main__':
    main()

