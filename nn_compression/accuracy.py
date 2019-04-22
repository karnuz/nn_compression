from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from vgg import VGG
#from vgg_shaokai import VGG as VGG_shaokai
#from convnet import ConvNet
#from resnet import ResNet18, ResNet50
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import admm
from testers import *
import time
from CIFAR10CUSTOM import CIFAR10CUSTOM

parser = argparse.ArgumentParser(description='PyTorch CIFAR training')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')
parser.add_argument('--sparsity_type', type=str, default='column',
                    help ="define sparsity_type: [irregular,column,filter]")
parser.add_argument('--config_file', type=str, default='config_vgg16',
                    help ="define sparsity_type: [irregular,column,filter]")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


#from admm import GradualWarmupScheduler
#from admm import CrossEntropyLossMaybeSmooth
#from admm import mixup_data, mixup_criterion


test_loader = torch.utils.data.DataLoader(
        CIFAR10CUSTOM('./data.cifar10/fivelabels', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

def test(model, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss = criterion(output, target)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def validate(val_loader,criterion, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()


    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # if config.gpu is not None:
            #     input = input.cuda(config.gpu, non_blocking=True)
            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      .format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

        print(' * Acc@1 {top1.avg:.3f} '
              .format(top1=top1))
        # global best_acc
        # if top1.avg.item()>best_acc and not config.admm:
        #     best_acc = top1.avg.item()
        #     print ('new best_acc is {top1.avg:.3f}'.format(top1=top1))
        #     print ('saving model {}'.format(config.save_model))
        #     torch.save(config.model.state_dict(),config.save_model)
        print('new best_acc is {top1.avg:.3f}'.format(top1=top1))

    return top1.avg



def main():
    model = VGG(depth=16, init_weights=True, cfg=None)
    # model = VGG_shaokai("vgg16")
    # model = ConvNet()
    # model = ResNet18()
    # model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("./model_pruned/2019-04-09 11:14:52.016169/column-filter-fivelabels-masked_retrain/cifar10_vgg16_retrained_acc_93.960_4rhos_config_vgg16_v2.yaml.pt"))
    model.cuda()

    criterion = F.cross_entropy
#    criterion = CrossEntropyLossMaybeSmooth(smooth_eps=0).cuda()
    validate(test_loader, criterion, model)
    # test(model, criterion, test_loader)
    print("\n------------------------------\n")


    print('here')
    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4 and "shortcut" not in name):
            print(name, weight.size())


    print('here now')
    test_column_sparsity(model)
    # test_chanel_sparsity(model)
    test_filter_sparsity(model)
    # test_irregular_sparsity(model)



    # for name, weight in model.named_parameters():
    #     if (len(weight.size()) == 4):
    #         print(weight.reshape(weight.shape[0], -1))


    # all_weight = []
    # for name, weight in model.named_parameters():
    #     if (len(weight.size()) == 4):
    #         print(np.shape(weight.cpu().detach().numpy()))
    #         temp = weight.cpu().detach().numpy().flatten()
    #         for item in temp:
    #             if item != 0:
    #                 all_weight.append(item)
    #
    #
    # yy = np.linspace(0, np.size(all_weight), np.size(all_weight), endpoint=False)
    #
    # plt.scatter(yy, all_weight, marker=".")
    # plt.show()
    # plt.hist(all_weight, bins=100, range=(-0.15, 0.15))
    # plt.show()




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
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



if __name__ == '__main__':
    main()
