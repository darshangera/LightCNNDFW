
'''
Extract features and generate similarity score

'''
from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import OrderedDict
import cv2
import numpy as np
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from light_cnn import LightCNN_9Layers,LightCNN_4Layers, LightCNN_29Layers, LightCNN_29Layers_v2
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch DFW Feature Extracting for simlarity generation')
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')
parser.add_argument('--cuda', '-c', default=False)
#parser.add_argument('--resume', default='./LightCNN_29_V2Layers_checkpoint.pth.tar', type=str, metavar='PATH',
parser.add_argument('--resume', default='./lightCNN_71_checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model', default='LightCNN-29v2', type=str, metavar='Model',
                    help='model type: LightCNN-9, LightCNN-29')
parser.add_argument('--root_path', default='../DFW2019/', type=str, metavar='PATH',
#parser.add_argument('--root_path', default='../DisguisedFacesInTheWild/', type=str, metavar='PATH',
                    help='root path of face images (default: none).')
#parser.add_argument('--img_list', default='../DisguisedFacesInTheWild/Testing_data_face_name.txt', type=str, metavar='PATH',
parser.add_argument('--img_list', default='../DFW2019/fileNames.txt', type=str, metavar='PATH',
                    help='list of face images for feature extraction (default: none).')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='save root path for features of face images.')
parser.add_argument('--num_classes', default=80013, type=int,
                    metavar='N', help='mini-batch size (default: 79077)')

def main():
    global args
    args = parser.parse_args()

    if args.model == 'LightCNN-9':
        model = LightCNN_9Layers(num_classes=args.num_classes)
    elif args.model == 'LightCNN-29':
        model = LightCNN_29Layers(num_classes=args.num_classes)
    elif args.model == 'LightCNN-4':
        model = LightCNN_4Layers(num_classes=args.num_classes)
    elif args.model == 'LightCNN-29v2':
        model = LightCNN_29Layers_v2(num_classes=args.num_classes)
    else:
        print('Error model type\n')

    model.eval()

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()


    if args.resume:
        if os.path.isfile(args.resume):
            if args.model =='LightCNN-4':
                pre_trained_dict = torch.load('./LightenedCNN_4_torch.pth', map_location ='cpu')# lambda storage, loc: storage)

                model_dict = model.state_dict()
                #model = model.to(device)  #lightcnn model
                pre_trained_dict['features.0.filter.weight'] = pre_trained_dict.pop('0.weight')
                pre_trained_dict['features.0.filter.bias'] = pre_trained_dict.pop('0.bias')
                pre_trained_dict['features.2.filter.weight'] = pre_trained_dict.pop('2.weight')
                pre_trained_dict['features.2.filter.bias'] = pre_trained_dict.pop('2.bias')
                pre_trained_dict['features.4.filter.weight'] = pre_trained_dict.pop('4.weight')
                pre_trained_dict['features.4.filter.bias'] = pre_trained_dict.pop('4.bias')
                pre_trained_dict['features.6.filter.weight'] = pre_trained_dict.pop('6.weight')

                my_dict = {k: v for k, v in pre_trained_dict.items() if ("fc2" not in k )}  #by DG

                model_dict.update(my_dict)
                model.load_state_dict(model_dict, strict = False)
            else:
                print("=> loading checkpoint '{}'".format(args.resume))
                #checkpoint = torch.load(args.resume, map_location='cpu')['state_dict']
                state_dict = torch.load(args.resume, map_location='cpu')['state_dict']#torch.load(directory, map_location=lambda storage, loc: storage)
                #state_dict = torch.load(args.resume, map_location=lambda storage, loc: storage)['state_dict']
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

    img_list  = read_list(args.img_list)
    #print(len(img_list))
    transform = transforms.Compose([transforms.ToTensor()])
    count     = 0
    input     = torch.zeros(1, 1, 128, 128)

    featuresmatrix = np.empty((0,256))

    for img_name in img_list[:]:
        img_name = img_name[0]
        count = count + 1
        img   = cv2.imread(os.path.join(args.root_path, img_name), cv2.IMREAD_GRAYSCALE)
        #print(os.path.join(args.root_path, img_name))
        #img   = cv2.imread(os.path.join(args.root_path, 'Cropped_'+img_name), cv2.IMREAD_GRAYSCALE)
        img =  cv2.resize(img,(128,128))
        img   = np.reshape(img, (128, 128, 1))
        img   = transform(img)
        input[0,:,:,:] = img

        start = time.time()
        '''
        if args.cuda:
            input = input.cuda()
        '''
        with torch.no_grad():
            input_var   = input#torch.tensor(input)#, volatile=True)
            _, features = model(input_var)
            #print(features.size())
            featuresmatrix =np.append(featuresmatrix , features.data.cpu().numpy(),axis = 0)
            #print(features)

        end  = time.time() - start
        #print("{}({}/{}). Time: {}".format(os.path.join(args.root_path, img_name), count, len(img_list), end))
        #save_feature(args.save_path, img_name, features.data.cpu().numpy()[0])
    #print(featuresmatrix.shape)
    similarity_matrix = cosine_similarity(featuresmatrix,featuresmatrix)
    #np.savetxt("similarity_score_validationset.txt",similarity_matrix,fmt ="%4.2f", delimiter=" ")
    np.savetxt("similarity_score_testset2019_lightcnn29_71.txt",similarity_matrix,fmt ="%5.4f", delimiter=" ")
    #similarity_matrix.tofile("similarity_score_testset2019.txt",sep=' ', format ='%4.2f')  #It gives single line not a matrix
    #pd.DataFrame(similarity_matrix).to_csv("similarity_score_testset2019.txt")

def read_list(list_path):
    img_list = []
    with open(list_path, 'r') as f:
        for line in f.readlines()[0:]:
            img_path = line.split('\n')
            img_list.append(img_path)
    print('There are {} images..'.format(len(img_list)))
    return img_list

def save_feature(save_path, img_name, features):
    img_path = os.path.join(save_path, img_name)
    img_dir  = os.path.dirname(img_path) + '/';
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    fname = os.path.splitext(img_path)[0]
    fname = fname + '.feat'
    fid   = open(fname, 'wb')
    fid.write(features)
    fid.close()

if __name__ == '__main__':
    main()
