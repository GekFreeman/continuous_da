# -*- coding:utf-8 -*-
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader
import resnet as models
import pdb
import hnswlib
import numpy as np
import time
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
pre_train_iters = 500
epochs = 10
batch_size = 32
iteration = 10000
lr = [0.001, 0.01]
LEARNING_RATE = 0.001
momentum = 0.9
cuda = True
seed = 8
log_interval = 10
l2_decay = 5e-4
root_path = "/root/share/Original_images/"
source1_name = "webcam"
source2_name = 'dslr'
target_name = "amazon"

dataset = "office31"

if dataset == "office31":
    source1_name = source1_name + "/images/"
    source2_name = source2_name + "/images/"
    target_name = target_name + "/images/"
    
# Target Hard
hard_threshold = 0.5
    
# HNSW
k = 100
ef = 60
ef_construction = 200
num_elements = 5000
dim = 256
M = 16
index = hnswlib.Index(space='l2', dim=dim) # dim 向量维度
    
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

source1_loader = data_loader.load_training(root_path, source1_name, batch_size, kwargs)
source2_loader = data_loader.load_training(root_path, source2_name, batch_size, kwargs)
target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)

source1_test_loader = data_loader.load_testing(root_path, source1_name, batch_size, kwargs)
target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)

def train(model):
    source1_iter = iter(source1_loader)
    correct = 0

    optimizer = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet2.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay) 
    
    for i in range(1, pre_train_iters + 1):
        model.train()

        optimizer.param_groups[0]['lr'] = lr[0] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[1]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[2]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[3]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        optimizer.param_groups[4]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        
        try:
            source_data, source_label = source1_iter.next()
        except Exception as err:
            source1_iter = iter(source1_loader)
            source_data, source_label = source1_iter.next()

        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        optimizer.zero_grad()

        cls_loss = model(source_data, label_src=source_label, mark=0)
        loss = cls_loss
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\t'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item()))

    correct = 0
    for epoch in range(epochs):
        # 更新hnsw
        index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
        index.set_ef(int(k * 1.2))

        start = time.time()
        source_nums, src_qry_label = test_hnsw(model, source1_test_loader, loader_type=1)
        tgt_spt_idx, tgt_qry_idx, tgt_spt_src, tgt_spt_src_prob, tgt_qry_label = test_hnsw(model, target_test_loader, loader_type=0, idx_init=source_nums, k=10)
        end = time.time()
        print("HNSW:", end - start)

        # meta-training dataloader
        tgt_idx, src_indices = index_generate_sim(tgt_idx=tgt_spt_idx, src_idx=tgt_spt_src, src_prob=tgt_spt_src_prob):
        pair1_target1_loader = data_loader.load_training_index(root_path, target_name, batch_size, tgt_idx, kwargs)
        pair1_source1_loader = data_loader.load_training_index(root_path, source1_name, batch_size, src_indices, kwargs)

        src_qry_idx1, src_qry_idx2 = index_generate_diff(src_indices, src_qry_label[src_indices])
        pair2_source1_loader1 = data_loader.load_training_index(root_path, target_name, batch_size, src_qry_idx1, kwargs)
        pair2_source1_loader2 = data_loader.load_training_index(root_path, source1_name, batch_size, src_qry_idx2, kwargs)

        cls_source1_loader = data_loader.load_training(root_path, source1_name, batch_size, kwargs)

        # meta-testing dataloader
        tgt_qry_idx1, tgt_qry_idx2 = index_generate_diff(tgt_qry_idx, tgt_qry_label)
        pair_3_target_loader1 = data_loader.load_training_index(root_path, target_name, batch_size, tgt_qry_idx1, kwargs)
        pair_3_target_loader2 = data_loader.load_training_index(root_path, target_name, batch_size, tgt_qry_idx2, kwargs)    


        pair1_iter_a = iter(pair1_target1_loader)
        pair1_iter_b = iter(pair1_source1_loader)

        pair2_iter_a = iter(pair2_source1_loader1)
        pair2_iter_b = iter(pair2_source1_loader2)

        trn_cls = iter(cls_source1_loader)

        pair3_iter_a = iter(pair_3_target_loader1)
        pair3_iter_b = iter(pair_3_target_loader2) 

        optimizer = torch.optim.SGD([
                {'params': model.sharedNet.parameters()},
                {'params': model.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
                {'params': model.cls_fc_son2.parameters(), 'lr': LEARNING_RATE},
                {'params': model.sonnet1.parameters(), 'lr': LEARNING_RATE},
                {'params': model.sonnet2.parameters(), 'lr': LEARNING_RATE},
            ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
        for it in (len(tgt_spt_idx)//batch_size):
            model.train()

            optimizer.param_groups[0]['lr'] = lr[0] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
            optimizer.param_groups[1]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
            optimizer.param_groups[2]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
            optimizer.param_groups[3]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
            optimizer.param_groups[4]['lr'] = lr[1] / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)

            pair1_data_a, pair1_label_a = pair1_iter_a.next()
            pair1_data_b, pair1_label_b = pair1_iter_b.next()
            pair2_data_a, pair2_label_a = pair2_iter_a.next()
            pair2_data_b, pair1_label_b = pair2_iter_b.next()
            source_data, source_label = trn_cls.next()
            
            
            
            if cuda:
                source_data, source_label = source_data.cuda(), source_label.cuda()
                target_data = target_data.cuda()
            source_data, source_label = Variable(source_data), Variable(source_label)
            target_data = Variable(target_data)
            optimizer.zero_grad()

            cls_loss, mmd_loss, l1_loss = model(source_data, target_data, source_label, mark=1)
            gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
            loss = cls_loss + gamma * (mmd_loss + l1_loss)
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))

            try:
                source_data, source_label = source2_iter.next()
            except Exception as err:
                source2_iter = iter(source2_loader)
                source_data, source_label = source2_iter.next()
            try:
                target_data, __ = target_iter.next()
            except Exception as err:
                target_iter = iter(target_train_loader)
                target_data, __ = target_iter.next()
            if cuda:
                source_data, source_label = source_data.cuda(), source_label.cuda()
                target_data = target_data.cuda()
            source_data, source_label = Variable(source_data), Variable(source_label)
            target_data = Variable(target_data)
            optimizer.zero_grad()

            cls_loss, mmd_loss, l1_loss = model(source_data, target_data, source_label, mark=2)
            gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
            loss = cls_loss + gamma * (mmd_loss + l1_loss)
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                print(
                    'Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                        i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))

            if i % (log_interval * 20) == 0:
                t_correct = test(model)
                if t_correct > correct:
                    correct = t_correct
                print(source1_name, source2_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")

def test_hnsw(model, data_loader, loader_type=1, idx_init=0, k=10):
    """
    loader_type: 1表示源域，0表示目标域
    """
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    tgt_spt_src = np.empty(shape=(0, k))
    tgt_spt_src_prob = np.empty(shape=(0, k))
    tgt_spt_idx = []
    tgt_qry_idx = []
    tgt_qry_label = []
    src_qry_label = []
    idx = idx_init
    with torch.no_grad():
        for data, target in data_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred1, feat = model(data, mark = -1)

            pred = torch.nn.functional.softmax(pred1, dim=1)

            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
            
            feat = feat.cpu().numpy()
            if loader_type == 0:
                tgt_hard = torch.where(pred.data.max(1)[0] < hard_threshold)[0].cpu().numpy()
                tgt_simple = torch.where(pred.data.max(1)[0] >= hard_threshold)[0].cpu().numpy()
                tgt_spt_idx = np.concatenate((tgt_spt_idx, [x + idx for x in tgt_hard]), axis=0)
                tgt_qry_idx = np.concatenate((tgt_qry_idx, [x + idx for x in tgt_simple]), axis=0)
                tgt_pred = pred.data.max(1)[1].cpu().numpy()
                tgt_qry_label = np.concatenate((tgt_qry_label, tgt_pred[tgt_simple]), axis=0)
                for spt_sample in tgt_hard:
                    sample_label = spt_sample + idx
                    index.add_items(feat[spt_sample], sample_label)
                    labels, distances = index.knn_query(feat[spt_sample], k=k)
                    tgt_spt_src = np.concatenate((tgt_spt_src, labels), axis=0)
                    tgt_spt_src_prob = np.concatenate((tgt_spt_src_prob, softmax(distances)), axis=0)
                    index.mark_deleted(sample_label)
            else:
                index.add_items(feat, [x + idx for x in range(len(data))])
                src_qry_label = np.concatenate([src_qry_label, target], axis=0)
            
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            idx += len(data)
        test_loss /= len(target_test_loader.dataset)
        print(target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))
        print('\nsource1 accnum {}, source2 accnum {}'.format(correct1, correct2))
    if loader_type == 1:
        return idx, src_qry_label
    else:
        return tgt_spt_idx, tgt_qry_idx, tgt_spt_src, tgt_spt_src_prob, tgt_qry_label
            
def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred1, pred2 = model(data, mark = 0)

            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)

            pred = (pred1 + pred2) / 2
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()

            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred2.data.max(1)[1]
            correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(target_test_loader.dataset)
        print(target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))
        print('\nsource1 accnum {}, source2 accnum {}'.format(correct1, correct2))
    return correct

def index_generate_sim(tgt_idx, src_idx, src_prob):
    src_indices = [np.random.choice(x, size=1, p=src_prob[i]) for i,x in enumerate(src_idx)]
    tgt_idx, src_indices = zip(*np.random.shuffle(zip(tgt_idx, src_indices)))
    return tgt_idx, src_indices


def index_generate_diff(tgt_qry_idx, tgt_qry_label):
    """
    优化：类别概率，样本概率
    """
    d = defaultdict(list)
    for s, l in zip(tgt_qry_idx, tgt_qry_label):
        d[l].append(s)
    category_list = list(d.keys())
    tgt_qry_idx_aux = [np.random.choice(d[random_chice_the_other(tgt_qry_label[i],categorys)], size=1) for i in len(tgt_qry_idx)]
    return zip(*np.random.shuffle(tgt_qry_idx, tgt_qry_idx_aux))
    
def random_chice_the_other(x, categorys):
    categorys = set(categorys)
    categorys.discard(x)
    return np.random.choice(list(categorys), size=1)
    

    
def softmax(x):
    """ softmax function """
    
    x -= np.max(x, axis = 1, keepdims = True) #为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    
    x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
    
    return x


if __name__ == '__main__':
    model = models.MFSAN(num_classes=31)
    if cuda:
        model.cuda()
    train(model)
