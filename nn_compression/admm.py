from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
from numpy import linalg as LA
import yaml
import math

class ADMM:
    def __init__(self, model, sparsity_type, prune_ratios, rho=0.001):
        self.ADMM_U = {}
        self.ADMM_Z = {}
        self.rho = rho
        self.rhos = {}
        self.sparsity_type = sparsity_type

        self.init(model,prune_ratios)

    def init(self, model,prune_ratios):
        """
        Args:
            config: configuration file that has settings for prune ratios, rhos
        called by ADMM constructor. config should be a .yaml file

        """
        self.prune_ratios = prune_ratios
        for k, v in self.prune_ratios.items():
            print(k,v)
            self.rhos[k] = self.rho
        for (name, W) in model.named_parameters():
            if(len(W.size()) == 4):
                if name not in self.prune_ratios:
                    continue
                self.ADMM_U[name] = torch.zeros(W.shape).cuda()  # add U
                self.ADMM_Z[name] = torch.Tensor(W.shape).cuda()  # add Z
                                    

                
                
def random_pruning(weight,sparsity_type, prune_ratio):
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    if (sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        indices = np.random.choice(shape2d[0], int(shape2d[0] * prune_ratio), replace=False)
        weight2d[indices, :] = 0
        weight = weight2d.reshape(shape)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = i not in indices
        weight = weight2d.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    else:
        raise Exception("not implemented yet")


def L1_pruning(sparsity_type, weight, prune_ratio):
    """
    projected gradient descent for comparison

    """
    percent = prune_ratio * 100
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy
    shape = weight.shape
    weight2d = weight.reshape(shape[0], -1)
    shape2d = weight2d.shape
    row_l1_norm = LA.norm(weight2d, 1, axis=1)
    percentile = np.percentile(row_l1_norm, percent)
    under_threshold = row_l1_norm < percentile
    above_threshold = row_l1_norm > percentile
    weight2d[under_threshold, :] = 0
    above_threshold = above_threshold.astype(np.float32)
    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    for i in range(shape2d[0]):
        expand_above_threshold[i, :] = above_threshold[i]
    weight = weight.reshape(shape)
    expand_above_threshold = expand_above_threshold.reshape(shape)
    return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()


def weight_pruning(weight,sparsity_type, prune_ratio):
    """
    weight pruning [irregular,column,filter]
    Args:
         weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
         prune_ratio (float between 0-1): target sparsity of weights

    Returns:
         mask for nonzero weights used for retraining
         a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero

    """

    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    percent = prune_ratio * 100
    if (sparsity_type == "irregular"):
        weight_temp = np.abs(weight)  # a buffer that holds weights with absolute values
        percentile = np.percentile(weight_temp, percent)  # get a value for this percentitle
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (sparsity_type == "column"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        column_l2_norm = LA.norm(weight2d, 2, axis=0)
        percentile = np.percentile(column_l2_norm, percent)
        under_threshold = column_l2_norm < percentile
        above_threshold = column_l2_norm > percentile
        weight2d[:, under_threshold] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[1]):
            expand_above_threshold[:, i] = above_threshold[i]
        expand_above_threshold = expand_above_threshold.reshape(shape)
        weight = weight.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        row_l2_norm = LA.norm(weight2d, 2, axis=1)
        percentile = np.percentile(row_l2_norm, percent)
        under_threshold = row_l2_norm < percentile
        above_threshold = row_l2_norm > percentile
        weight2d[under_threshold, :] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = above_threshold[i]
        weight = weight.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (sparsity_type == "bn_filter"):
        ## bn pruning is very similar to bias pruning
        weight_temp = np.abs(weight)
        percentile = np.percentile(weight_temp, percent)
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()
    else:
        raise SyntaxError("Unknown sparsity type")


def hard_prune(ADMM, model, option=None):
    """
    hard_pruning, or direct masking
    Args:
         model: contains weight tensors in cuda

    """
    
    print("hard pruning")
    mask = {}
    for (name, W) in model.named_parameters():
        if name not in ADMM.prune_ratios:  # ignore layers that do not have rho
            continue
        cuda_pruned_weights = None
        if option == None:
            layer_mask, cuda_pruned_weights = weight_pruning(W,ADMM.sparsity_type, ADMM.prune_ratios[name])  # get sparse model in cuda

        elif option == "random":
            layer_mask, cuda_pruned_weights = random_pruning(W,ADMM.sparsity_type, ADMM.prune_ratios[name])

        elif option == "l1":
            layer_mask, cuda_pruned_weights = L1_pruning(W,ADMM.sparsity_type, ADMM.prune_ratios[name])
        else:
            raise Exception("not implmented yet")
        mask[name]=layer_mask
        W.data = cuda_pruned_weights  # replace the data field in variable
    return mask

def test_sparsity(ADMM, model):
    """
    test sparsity for every involved layer and the overall compression rate

    """
    total_zeros = 0
    total_nonzeros = 0
    if ADMM.sparsity_type == "irregular":
        for i, (name, W) in enumerate(model.named_parameters()):
            if 'bias' in name:
                continue
            W = W.cpu().detach().numpy()
            zeros = np.sum(W == 0)
            total_zeros += zeros
            nonzeros = np.sum(W != 0)
            total_nonzeros += nonzeros
            print("sparsity at layer {} is {}".format(name, zeros / (zeros + nonzeros)))
        total_weight_number = total_zeros + total_nonzeros
        print('overal compression rate is {}'.format(total_weight_number / total_nonzeros))
    elif ADMM.sparsity_type == "column":
        for i, (name, W) in enumerate(model.named_parameters()):

            if 'bias' in name or name not in ADMM.prune_ratios:
                continue
            W = W.cpu().detach().numpy()
            shape = W.shape
            W2d = W.reshape(shape[0], -1)
            column_l2_norm = LA.norm(W2d, 2, axis=0)
            zero_column = np.sum(column_l2_norm == 0)
            nonzero_column = np.sum(column_l2_norm != 0)
            total_zeros += np.sum(W == 0)
            total_nonzeros += np.sum(W != 0)
            print("column sparsity of layer {} is {}".format(name, zero_column / (zero_column + nonzero_column)))
        print(
            'only consider conv layers, compression rate is {}'.format((total_zeros + total_nonzeros) / total_nonzeros))
    elif ADMM.sparsity_type == "filter":
        for i, (name, W) in enumerate(model.named_parameters()):
            if 'bias' in name or name not in ADMM.prune_ratios:
                continue
            W = W.cpu().detach().numpy()
            shape = W.shape
            W2d = W.reshape(shape[0], -1)
            row_l2_norm = LA.norm(W2d, 2, axis=1)
            zero_row = np.sum(row_l2_norm == 0)
            nonzero_row = np.sum(row_l2_norm != 0)
            total_zeros += np.sum(W == 0)
            total_nonzeros += np.sum(W != 0)
            print("filter sparsity of layer {} is {}".format(name, zero_row / (zero_row + nonzero_row)))
        print(
            'only consider conv layers, compression rate is {}'.format((total_zeros + total_nonzeros) / total_nonzeros))
    elif ADMM.sparsity_type == "bn_filter":
        for i, (name, W) in enumerate(model.named_parameters()):
            if name not in ADMM.prune_ratios:
                continue
            W = W.cpu().detach().numpy()
            zeros = np.sum(W == 0)
            nonzeros = np.sum(W != 0)
            print("sparsity at layer {} is {}".format(name, zeros / (zeros + nonzeros)))


def admm_initialization(args, ADMM, model):
    if not args.admm:
        return
    for i, (name, W) in enumerate(model.named_parameters()):
        if name in ADMM.prune_ratios:
            _, updated_Z = weight_pruning(W,ADMM.sparsity_type, ADMM.prune_ratios[name])  # Z(k+1) = W(k+1)+U(k)  U(k) is zeros her
            ADMM.ADMM_Z[name] = updated_Z


def z_u_update(args, ADMM, model, device, train_loader, optimizer, epoch, data, batch_idx, writer):
    if not args.admm:
        return

    if epoch != 1 and (epoch - 1) % args.admm_epoch == 0 and batch_idx == 0:
        for i, (name, W) in enumerate(model.named_parameters()):
            if name not in ADMM.prune_ratios:
                continue
            Z_prev = None
            if (args.verbose):
                Z_prev = torch.Tensor(ADMM.ADMM_Z[name].cpu()).cuda()
            ADMM.ADMM_Z[name] = W + ADMM.ADMM_U[name]  # Z(k+1) = W(k+1)+U[k]

            _, updated_Z = weight_pruning(ADMM.ADMM_Z[name], ADMM.sparsity_type,
                                          ADMM.prune_ratios[name])  # equivalent to Euclidean Projection
            ADMM.ADMM_Z[name] = updated_Z
            if (args.verbose):
                if writer:
                    writer.add_scalar('layer:{} W(k+1)-Z(k+1)'.format(name),
                                      torch.sqrt(torch.sum((W - ADMM.ADMM_Z[name]) ** 2)).item(), epoch)
                    writer.add_scalar('layer:{} Z(k+1)-Z(k)'.format(name),
                                      torch.sqrt(torch.sum((ADMM.ADMM_Z[name] - Z_prev) ** 2)).item(), epoch)
                # print ("at layer {}. W(k+1)-Z(k+1): {}".format(name,torch.sqrt(torch.sum((W-ADMM.ADMM_Z[name])**2)).item()))
                # print ("at layer {}, Z(k+1)-Z(k): {}".format(name,torch.sqrt(torch.sum((ADMM.ADMM_Z[name]-Z_prev)**2)).item()))
            ADMM.ADMM_U[name] = W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name]  # U(k+1) = W(k+1) - Z(k+1) +U(k)


def append_admm_loss(args, ADMM, model, ce_loss):
    '''
    append admm loss to cross_entropy loss
    Args:
        args: configuration parameters
        model: instance to the model class
        ce_loss: the cross entropy loss
    Returns:
        ce_loss(tensor scalar): original cross enropy loss
        admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
        ret_loss(scalar): the mixed overall loss

    '''
    admm_loss = {}

    if args.admm:

        for i, (name, W) in enumerate(model.named_parameters()):  ## initialize Z (for both weights and bias)
            if name not in ADMM.prune_ratios:
                continue

            admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm(W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name], p=2) ** 2)
    mixed_loss = 0
    mixed_loss += ce_loss
    for k, v in admm_loss.items():
        mixed_loss += v
    return ce_loss, admm_loss, mixed_loss


def masked_retrain(args, ADMM, model, device, train_loader, optimizer, epoch, masks):
    if not args.masked_retrain:
        return

    model.train()

    
    '''
    masks = {}
    for i, (name, W) in enumerate(model.named_parameters()):
        if name not in ADMM.prune_ratios:
            continue
        above_threshold, W = weight_pruning(args, W, ADMM.prune_ratios[name])
        W.data = W
        masks[name] = above_threshold
    '''

    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)

        loss.backward()

        for i, (name, W) in enumerate(model.named_parameters()):
            for mask in masks:
                if name in mask:
                    W.grad *= mask[name]

        optimizer.step()
        #quantize
#        with torch.no_grad():
#            quantize_linear_fix_zeros(model)
        
        if batch_idx % args.log_interval == 0:
            print("({}) cross_entropy loss: {}".format(args.optmzr, loss))
            print('re-Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    # admm.test_sparsity(args, ADMM, model)


def quantize_fixed(model, bit_length=8,bit_length_integer=0, **unused):

    mul_coeff =2**(bit_length-1-bit_length_integer)
    div_coeff =2**(bit_length_integer - bit_length + 1)
    max_coeff =2**(bit_length-1)
    
    for param_name, param in model.named_parameters():
        if param.dim()>1:
            param.mul_(mul_coeff).floor_().clamp_(-max_coeff - 1, max_coeff - 1).mul_(div_coeff)


def quantize_linear_fix_zeros(model,k=256):
    magic_percentile = 0.001
    for param_name,param in model.named_parameters():
        zero_mask = torch.eq(param, 0.0)  # get zero mask
        num_param = param.numel()
        kth = int(math.ceil(num_param * magic_percentile))
        param_flatten = param.view(num_param)
        param_min, _ = torch.topk(param_flatten, kth, dim=0, largest=False, sorted=False)
        param_min = param_min.max()
        param_max, _ = torch.topk(param_flatten, kth, dim=0, largest=True, sorted=False)
        param_max = param_max.min()
        step = (param_max - param_min) / (k - 2)
        param.clamp_(param_min, param_max).sub_(param_min).div_(step).round_().mul_(step).add_(param_min)
        param.masked_fill_(zero_mask, 0)  # recover zeros
