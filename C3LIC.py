import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import os
import torch.utils.data as data
from dataset_fine import dataset
import random

from model.resnet50 import ResNet50
from config import getConfig
from train_model import train_model


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    config = getConfig()
    print(config)

    os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_ids

    set_seed(config.random_seed)

    #---------------------------- dataset ------------------------------------
    if config.dataset == 'cub_bird':
        classes = 200
        data_dir = '/home/dell/data/dataset/CUB_200_2011'
    elif config.dataset == 'stanford_dog':
        classes = 120
        data_dir = '/home/dell/data/dataset/Dog/'
    elif config.dataset == 'aircraft':
        classes = 100
        data_dir = '/home/dell/data/dataset/fgvc_aircraft_2013b/data/'
    elif config.dataset == 'nabirds':
        classes = 555
        data_dir = '/home/dell/data/dataset/nabirds/'
    elif config.dataset == 'vegfru':
        classes = 292
        data_dir = './datasets/vegfru/'
    else:
        print('undefined dataset ! ')
    print('Dataset:', config.dataset, ', num of classes:', classes)
    print(data_dir)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomCrop((224,224)),          
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_set = dataset(config.dataset, root_dir=data_dir, transforms = data_transforms['train'], train =True)
    print('train_set',len(train_set))
    val_set = dataset(config.dataset ,root_dir=data_dir, transforms = data_transforms['val'], test =True)
    print('val_set',len(val_set))
    base_set = dataset(config.dataset, root_dir=data_dir, transforms = data_transforms['val'], train =True)
    print('basa_set',len(base_set))

    dataloader = {}
    dataloader['train']= data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    setattr(dataloader['train'], 'total_item_len', len(train_set))
    dataloader['val'] = data.DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    setattr(dataloader['val'], 'total_item_len', len(val_set))
    dataloader['base'] = data.DataLoader(base_set, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    setattr(dataloader['base'], 'total_item_len', len(base_set))

    # ---------------------------- model ------------------------------------
    if config.model_name == 'resnet50':
        model = ResNet50(classes)
    else:
        print('undefined model ! ')

    model = nn.DataParallel(model)
    model.cuda()

    # ---------------------------- loss and opt ------------------------------------
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # ---------------------------- log and train ------------------------------------
    log_file = open(config.model_name+'_'+config.dataset+'_'+str(config.code_length)+'.log', 'a')
    log_file.write(str(config))
    log_file.write('\n')
    print('training start ...')
    train_model(model, dataloader, optimizer,exp_lr_scheduler, config.epoch, config.code_length,log_file,config.theta,config.tao)
    log_file.close()





