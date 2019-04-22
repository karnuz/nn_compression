from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from vgg import VGG
import numpy as np
import admm
import os
import sys
from CIFAR10CUSTOM import CIFAR10CUSTOM
from shutil import copyfile
import datetime
import yaml

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 admm training')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=80, metavar='N',
                    help='number of epochs to train (default: 80)')
parser.add_argument('--optmzr', type=str, default='adam', metavar='OPTMZR',
                    help='optimizer used (default: adam)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--lr_num', type=int, default=3, metavar='LR_num',
                    help='number of learning rate (default: 3)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_model', type=str, default="",
                    help='For Saving the current Model')
parser.add_argument('--load_model', type=str, default=None,
                    help='For loading the model')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')
parser.add_argument('--load-model', action='store_true', default=False,
                    help='For loading the model')
parser.add_argument('--masked_retrain', action='store_true', default=True,
                    help='for masked retrain')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='whether to report admm convergence condition')
parser.add_argument('--admm', action='store_true', default=True,
                    help="for admm training")
parser.add_argument('--admm_epoch', type=int, default=1,
                    help="how often we do admm update")
parser.add_argument('--rho', type=float, default = 0.0001,
                    help ="define rho for ADMM")
parser.add_argument('--rho_num', type=int, default = 4,
                    help ="define how many rohs for ADMM training")
parser.add_argument('--sparsity_type', type=str, default='filter',
                    help ="define sparsity_type: [irregular,column,filter]")
parser.add_argument('--config_file', type=str, default='config_vgg16_v2.yaml',
                    help ="define sparsity_type: [irregular,column,filter]")
parser.add_argument('--no_Of_Labels', type = int, default=10,
                    help="provide number of labels")
parser.add_argument('--sparsity_list', type=str, default='column filter',
                    help ="define list of prunings separated by a space")
parser.add_argument('--base_model_path', type=str, default='./model/cifar10_vgg16_acc_93.700_3fc_sgd_bnin.pt',
                    help = "give the base model path")
parser.add_argument('--quantization', type=str, default=True,
                                        help = "do quantization ?")

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
writer = None


def train(args, ADMM, model, device, train_loader, optimizer, epoch, writer, masks):
    model.train()

    #print(masks)
    ce_loss = None
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        ce_loss = F.cross_entropy(output, target)

        admm.z_u_update(args, ADMM, model, device, train_loader, optimizer, epoch, data, batch_idx, writer)  # update Z and U variables
        ce_loss, admm_loss, mixed_loss = admm.append_admm_loss(args, ADMM, model, ce_loss)  # append admm losss

        mixed_loss.backward()

        
        for name, W in model.named_parameters():
            for mask in masks:
                if name in mask:
                    W.grad *= mask[name]

        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print("({}) cross_entropy loss: {}, mixed_loss : {}".format(args.optmzr, ce_loss, mixed_loss))
            print('admm Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), ce_loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return (100. * correct / len(test_loader.dataset))



def getData(n):

    if n == 2:
        subpath = '/twolabels'
    if n == 5:
        subpath = '/fivelabels'
    if n == 10:
        subpath = '/tenlabels'
                        
    train_loader = torch.utils.data.DataLoader(
        CIFAR10CUSTOM('./data.cifar10'+subpath, train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        CIFAR10CUSTOM('./data.cifar10'+subpath, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


def main():
    date = str(datetime.datetime.now())
    
    torch.cuda.set_device(0)
    train_loader, test_loader = getData(args.no_Of_Labels)
        
    if args.depth == 16:
        model = VGG(depth=16, init_weights=True, cfg=None)
    if args.depth == 19:
        model = VGG(depth=19, init_weights=True, cfg=None)
                
    admm = False
    masked_retrain = False

    if args.admm:
        admm = True
    if args.masked_retrain:
        masked_retrain = True
    else:
        print("no sparsity type specified")
        return


    config = args.config_file
    prune_ratios = []
    if not isinstance(config, str):
        raise Exception("filename must be a str")
    with open(config, "r") as stream:
        try:
            raw_dict = yaml.load(stream)
            prune_ratios = raw_dict['prune_ratios']  #this should be a list of dictionaries
            
        except yaml.YAMLError as exc:
            print(exc)

    masks = []
    base_model_path = args.base_model_path

    pruned_path = "./model_pruned/"+date+"/"
    if not os.path.exists(pruned_path):
        os.makedirs(pruned_path)
    copyfile(config, pruned_path+config)
    
    for i in range(len(prune_ratios)):
        prune_ratio = prune_ratios[i]
        print(prune_ratio)
        sparsity_type = prune_ratio['prune_ratio_'+str(i+1)]['type']
        prune_values = prune_ratio['prune_ratio_'+str(i+1)]['values']

        pruned_path = "./model_pruned/"+date+"/"
        for j in range(i+1):
            pruned_path += prune_ratios[j]['prune_ratio_'+str(j+1)]['type']
            pruned_path += '-'
        if args.no_Of_Labels == 2:
            pruned_path += 'twolabels-'
        if args.no_Of_Labels == 5:
            pruned_path += 'fivelabels-'
        if args.no_Of_Labels == 10:
            pruned_path += 'tenlabels-'

        admm_path = pruned_path +'admm'
        masked_path = pruned_path + 'masked_retrain'

        if not os.path.exists(admm_path):
                os.makedirs(admm_path)
        if not os.path.exists(masked_path):
                os.makedirs(masked_path)

        
        if args.admm and args.masked_retrain:
            saved_model = do_admmtrain(args,model,train_loader,test_loader,sparsity_type,prune_values, masks,base_model_path,admm_path)
            masked_model, mask = do_masked_retrain(args,model,train_loader,test_loader,sparsity_type,prune_values,masks,saved_model,masked_path)
            #masks.append(mask)
            base_model_path = masked_model
        elif args.admm:
            do_admmtrain(args,model,train_loader,test_loader,sparsity_type,prune_values,masks, base_model, admm_path)
        elif args.masked_retrain:
            do_masked_retrain(args,model, trained_loader, test_loader, sparsity_type, prune_values ,masks,saved_model,masked_path)
        else:
            print('error')
            

            
def do_admmtrain(args,model,train_loader,test_loader,sparsity_type,prune_ratios,masks,base_model_path,admm_path):
    """====================="""
    """ multi-rho admm train"""
    """====================="""
    initial_rho = args.rho
    current_rho = initial_rho
    
    if args.admm:
        for i in range(args.rho_num):
            current_rho = initial_rho * 10 ** i
            if i == 0:
                print("Loading" + base_model_path)
                model.load_state_dict(torch.load(base_model_path)) # admm train need basline model
                model.cuda()
            else:
                print("Loading: "+admm_path+"/cifar_vgg{}_{}_{}_{}.pt".format(args.depth, current_rho / 10, args.config_file, args.optmzr))
                model.load_state_dict(torch.load(admm_path+"/cifar_vgg{}_{}_{}_{}.pt".format(args.depth, current_rho / 10, args.config_file, args.optmzr)))
                model.cuda()
                
                
            ADMM = admm.ADMM(model, sparsity_type,prune_ratios, rho = current_rho)
            admm.admm_initialization(args, ADMM=ADMM, model=model)  # intialize Z variable
            
            # admm train
            best_prec1 = 0.
            lr = args.lr / 10
            if args.optmzr == "adam":
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
            if args.optmzr == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
            for epoch in range(1, args.epochs + 1):
                print("current rho: {}".format(current_rho))
                train(args, ADMM, model, device, train_loader, optimizer, epoch, writer,masks)

                prec1 = test(args, model, device, test_loader)
                best_prec1 = max(prec1, best_prec1)
                
            print("Best Acc: {:.4f}".format(best_prec1))
            print("Saving model: " + admm_path+"/cifar_vgg{}_{}_{}_{}.pt".format(args.depth, current_rho, args.config_file, args.optmzr))
            torch.save(model.state_dict(), admm_path+"/cifar_vgg{}_{}_{}_{}.pt".format(args.depth, current_rho, args.config_file, args.optmzr))

    return admm_path+"/cifar_vgg{}_{}_{}_{}.pt".format(args.depth, current_rho, args.config_file, args.optmzr)
            
"""========================"""
"""END multi-rho admm train"""
"""========================"""



def do_masked_retrain(args,model,train_loader,test_loader,sparsity_type,prune_ratios, masks,base_model, masked_path):
    """=============="""
    """masked retrain"""
    """=============="""

    initial_rho = args.rho
    current_rho = initial_rho

    if args.masked_retrain:
        # load admm trained model
        print("Loading: " + base_model)
        model.load_state_dict(torch.load(base_model))
        model.cuda()
            
        if args.optmzr == "adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.optmzr == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
        
        ADMM = admm.ADMM(model, sparsity_type, prune_ratios, rho = initial_rho) # rho doesn't matter here
        best_prec1 = [0]
        mask = admm.hard_prune(ADMM, model)
        masks.append(mask)
        saved_model_path = ''

        for epoch in range(1, args.epochs*2 + 1):
            scheduler.step()
            admm.masked_retrain(args, ADMM, model, device, train_loader, optimizer, epoch,masks)

            prec1 = test(args, model, device, test_loader)
            if prec1 > max(best_prec1):
                print("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n".format(prec1))
                print("Saving Model: "+ masked_path+"/cifar10_vgg{}_retrained_acc_{:.3f}_{}rhos_{}.pt".format(args.depth, prec1, args.rho_num, args.config_file))
                torch.save(model.state_dict(), masked_path+"/cifar10_vgg{}_retrained_acc_{:.3f}_{}rhos_{}.pt".format(args.depth, prec1, args.rho_num, args.config_file))
                saved_model_path = masked_path+"/cifar10_vgg{}_retrained_acc_{:.3f}_{}rhos_{}.pt".format(args.depth, prec1, args.rho_num, args.config_file)
                print("\n>_ Deleting previous model file with accuracy {:.3f}% now...\n".format(max(best_prec1)))
                if len(best_prec1) > 1:
                    os.remove(masked_path+"/cifar10_vgg{}_retrained_acc_{:.3f}_{}rhos_{}.pt".format(args.depth, max(best_prec1), args.rho_num, args.config_file))
                    
                    
            best_prec1.append(prec1)
                    
        admm.test_sparsity(ADMM, model)
                    
        print("Best Acc: {:.4f}".format(max(best_prec1)))
        return saved_model_path,mask     
                    
        """=============="""
        """masked retrain"""
        """=============="""
                    
                                        
'''                                        
                                        
def main():
    torch.cuda.set_device(0)
    if (args.admm and args.masked_retrain):
        raise ValueError("can't do both masked retrain and admm")

    if args.no_Of_Labels == 2:
        subpath = '/twolabels'
    if args.no_Of_Labels == 5:
        subpath = '/fivelabels'
    
    train_loader = torch.utils.data.DataLoader(
        CIFAR10CUSTOM('./data.cifar10'+subpath, train=True, download=False,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        CIFAR10CUSTOM('./data.cifar10'+subpath, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    if args.depth == 16:
        model = VGG(depth=16, init_weights=True, cfg=None)
    if args.depth == 19:
        model = VGG(depth=19, init_weights=True, cfg=None)


    """====================="""
    """ multi-rho admm train"""
    """====================="""
    initial_rho = args.rho

    if args.admm:
        for i in range(args.rho_num):
            current_rho = initial_rho * 10 ** i
            if i == 0:
                model.load_state_dict(torch.load("./model/cifar10_vgg16_acc_93.700_3fc_sgd_bnin.pt".format(args.depth))) # admm train need basline model
                # model = nn.DataParallel(model)
                # model.to(device)
                model.cuda()
            else:
                model.load_state_dict(torch.load("./model_prunned"+subpath+"/cifar10_vgg{}_{}_{}_{}.pt".format(args.depth, current_rho / 10, args.config_file, args.optmzr)))
                # model = nn.DataParallel(model)
                # model.to(device)
                model.cuda()


            ADMM = admm.ADMM(model, file_name=args.config_file+".yaml", rho=current_rho)
            admm.admm_initialization(args, ADMM=ADMM, model=model)  # intialize Z variable

            # admm train
            best_prec1 = 0.
            lr = args.lr / 10
            if args.optmzr == "adam":
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
            if args.optmzr == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
            # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
            for epoch in range(1, args.epochs + 1):
                print("current rho: {}".format(current_rho))
                train(args, ADMM, model, device, train_loader, optimizer, epoch, writer)
                prec1 = test(args, model, device, test_loader)
                best_prec1 = max(prec1, best_prec1)

            # if args.sparsity_type == "filter":
            #     admm.hard_prune(args, ADMM, model)

            print("Best Acc: {:.4f}".format(best_prec1))
            print("Saving model...")
            torch.save(model.state_dict(), "./model_prunned"+subpath+"/cifar10_vgg{}_{}_{}_{}.pt".format(args.depth, current_rho, args.config_file, args.optmzr))

    """========================"""
    """END multi-rho admm train"""
    """========================"""




    """=============="""
    """masked retrain"""
    """=============="""

    if args.masked_retrain:
        # load admm trained model
        print("\n>_ Loading file: ./model_prunned"+subpath+"/cifar10_vgg{}_{}_{}_{}.pt".format(args.depth, initial_rho*10**(args.rho_num-1), args.config_file, args.optmzr))
        model.load_state_dict(torch.load("./model_prunned"+subpath+"/cifar10_vgg{}_{}_{}_{}.pt".format(args.depth, initial_rho*10**(args.rho_num-1), args.config_file, args.optmzr)))
        # model = nn.DataParallel(model)
        # model.to(device)
        model.cuda()


        if args.lr_num == 3:
            l            if prec1 > max(best_prec1):
                print("\n>_ Got better accuracy, saving model with accuracy {:.3f}% now...\n".format(prec1))
                torch.save(model.state_dict(), "./model_retrained"+subpath+"/cifar10_vgg{}_retrained_acc_{:.3f}_{}rhos_{}.pt".format(args.depth, prec1, args.
'''

if __name__ == '__main__':
    main()
