import torch
import torch.nn as nn
import cv2
import math
import random
import numpy as np
from torch.autograd import Variable 
import torch.nn.functional as F


class AverageMeter(object):
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


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()



def cosine_contrastive_loss(x1, x2, y):
    """
    Compute the cosine contrastive loss for a batch of pairs.
    :param x1: tensor of shape (batch_size, embedding_size)
    :param x2: tensor of shape (batch_size, embedding_size)
    :param y: tensor of shape (batch_size,), with elements 0 or 1 indicating whether the pairs are similar or dissimilar
    :param margin: margin hyperparameter controlling the minimum distance between embeddings of dissimilar pairs
    :return: scalar tensor representing the average loss over the batch
    """
    margin = 1.0
    cos_sim = F.cosine_similarity(x1, x2)
    loss = (1 - y) * F.relu(margin - cos_sim).pow(2) + y * cos_sim.pow(2)
    return loss.mean()

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    # y_pred_softmax = torch.sigmoid(y_pred)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

def multi_acc_multi_label(y_pred,y_test):
    outputs = torch.sigmoid(y_pred).cpu()
    y_test = y_test.cpu()
    # predicted = torch.floor(outputs + 0.5)

    predicted = torch.round(outputs.float())

    correct = (predicted == y_test).float()

    # print(predicted)
    # print(y_test)

    # print(correct)

    # print(correct.sum())


    correct = (correct.sum(dim=1)/correct.size(1)).sum()

    acc = correct / y_test.size(0)

    return torch.round(acc * 100)

# y_pred = torch.Tensor([[-0.1,0.6,0.7,-0.1],[0.7,-0.1,0.0,0.0]])
# y_test = torch.Tensor([[0.0,1.0,1.0,0.0],[1.0,0.0,0.0,0.0]])

# y_pred = torch.Tensor([[-0.1,0.6,0.7,-0.1]])
# y_test = torch.Tensor([[0.0,1.0,1.0,0.0]])
# #
# print(multi_acc_multi_label(y_pred, y_test))

# y_pred_sigmoid = torch.sigmoid(y_pred)

# print(y_pred_sigmoid)

from sklearn.metrics import f1_score


def get_f1(y_pred, y_true):
    # print(y_pred.shape)
    # print(y_true.shape)
    # print(y_pred)
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    # print(y_pred_softmax)
    # y_pred_softmax = torch.sigmoid(y_pred)
    _, y_pred = torch.max(y_pred_softmax, dim = 1) 
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()

    # print(y_pred)
    # print(y_true)
    
    return f1_score(y_true, y_pred)

def get_f1_multi_label(y_pred, y_true):

    outputs = torch.sigmoid(y_pred).cpu()

    predicted = torch.round(outputs.float())

    y_pred = predicted.flatten().detach().cpu()
    y_true = y_true.flatten().detach().cpu()

    return f1_score(y_true, y_pred)

# print(get_f1_multi_label(y_pred,y_test))


def cyclical_lr(stepsize, max_lr=3e-4, min_lr=3e-3):
    
    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.
    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)
    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda


def get_class_distribution(obj):
	
    count_dict = {
        "non_defect": 0,
        "defect": 0,
    }
    
    for i in obj:
        if i == 0: 
            count_dict["non_defect"] += 1
        elif i == 1: 
            count_dict["defect"] += 1        
        else:
            print("Check classes.")
    
    print(count_dict)
            
    return count_dict

def get_indices_synthetic(df, num_epochs, AA_cut, A_cut, B_cut, C_cut):
    
    AA_syn_all_indices = df.index[(df['cumulative_grade'] == 'AA') & (df['org_syn_flag'] == 0)].tolist()
    AA_syn_indices = random.sample(AA_syn_all_indices, num_epochs*AA_cut)
    
    A_syn_all_indices = df.index[(df['cumulative_grade'] == 'A') & (df['org_syn_flag'] == 0)].tolist()
    A_syn_indices = random.sample(A_syn_all_indices, num_epochs*A_cut)
    
    B_syn_all_indices = df.index[(df['cumulative_grade'] == 'B') & (df['org_syn_flag'] == 0)].tolist()
    B_syn_indices = random.sample(B_syn_all_indices, num_epochs*B_cut) 
    
    C_syn_all_indices = df.index[(df['cumulative_grade'] == 'C') & (df['org_syn_flag'] == 0)].tolist()
    C_syn_indices = random.sample(C_syn_all_indices, num_epochs*C_cut)
    
    return AA_syn_indices, A_syn_indices, B_syn_indices, C_syn_indices


def get_class_wts(df, label_dataset):

    target_list = []

    for t in label_dataset:

        target_list.append(t)

    target_list = torch.tensor(target_list)

    class_count = [i for i in get_class_distribution(df['label']).values()]

    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 

    class_weights_all = class_weights[target_list]

    return class_weights_all


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



class InterclassLoss(nn.Module):
    def __init__(self):
        super(InterclassLoss, self).__init__()

    def forward(self, pred, truth):
        loss = 0.0
        # loss = Variable(loss.data, requires_grad=True)
        pred = pred.type(torch.FloatTensor)
        _, predicted = torch.max(pred.data, 1)
        
        predicted = torch.Tensor(predicted.type(torch.FloatTensor))
        predicted = predicted.cuda()
        pred = pred.cuda()
        
        for i, p in enumerate(pred):
            for j in range(0,4):
                if j == truth[i]:
                    M = self._getM(truth[i]).cuda()
                    weight = (abs(predicted[i]-truth[i])+1)/(M.cuda())
                    loss += weight*(-p[j])
        loss = loss/len(pred)
        loss = Variable(loss, requires_grad=True)
        return loss

    
    def _getM(self,label):
        M = 0.0
        for i in range(0,4):
            M += abs(label-i)+1
        return M
