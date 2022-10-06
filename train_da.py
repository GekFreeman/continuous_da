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
from tensorboardX import SummaryWriter
import utils
import utils.optimizers as optimizers
import models
import yaml
from tqdm import tqdm
import torch.nn as nn
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--s1", type=str)
parser.add_argument("--s2", type=str)
parser.add_argument("--t", type=str)
parser.add_argument("--mode", type=int)
parser.add_argument("--seed", type=int)
parser.add_argument("--threshold", type=float)
parser.add_argument("--on_lbd", type=float)
parser.add_argument("--push_scale", type=float)
args = parser.parse_args()

#import tensorflow as tf

num_gpu=[2, 3]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in num_gpu)

# Training settings
batch_size = 64
iteration = 10000
lr = [0.001, 0.01]
lr_scale=[1000,1]
LEARNING_RATE = 0.001
momentum = 0.9
cuda = True
seed = args.seed
log_interval = 10
l2_decay = 5e-4
root_path ="/userhome/root/share/image-clef-proc/" # 12
source1_name = args.s1
source2_name = args.s2
sources = [source1_name, source2_name]
original_target_name = args.t
dataset = "Image-CLEF"
    
# Target Hard
hard_threshold = args.threshold
    
# HNSW
k = 100
ef = 60
ef_construction = 200
num_elements = 5000
dim = 512
M = 16
    
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
np.random.seed(seed)
    
kwargs = {'num_workers':4, 'pin_memory': True} if cuda else {}

def pretrain(model, optimizer, source_name, target_name, lr_scheduler, log=False, epochs=10, writer=None):
    new_source_name = source_name
    new_target_name = target_name
    if dataset == "office31":
        new_source_name = source_name + '/images/'
        new_target_name = target_name + '/images/'
    source1_loader = data_loader.load_training(root_path, new_source_name, batch_size, kwargs)
    target_loader = data_loader.load_training(root_path, new_target_name, batch_size, kwargs)
    
    source1_test_loader = data_loader.load_testing(root_path, new_source_name, batch_size, kwargs)
    target_test_loader = data_loader.load_testing(root_path, new_target_name, batch_size, kwargs)
    iters = 0
    iteration = len(source1_loader)
    iter_target_loader = iter(target_loader)
    for ep in range(epochs):
        for i, data in enumerate(source1_loader):
            optimizer.param_groups[0]['lr'] = lr[0] / math.pow((1 + 10 * (i + iters) / (iteration * epochs)), 0.75)
            optimizer.param_groups[1]['lr'] = lr[1] / math.pow((1 + 10 * (i + iters) / (iteration * epochs)), 0.75)
            optimizer.param_groups[2]['lr'] = lr[1] / math.pow((1 + 10 * (i + iters) / (iteration * epochs)), 0.75)
            source_data, source_label = data
            
            target_data, target_label, iter_target_loader = save_iter(iter_target_loader, target_loader)
            
            optimizer.zero_grad()

            cls_loss, mmd_loss = model(None,
                   [source_data, source_label, target_data, target_label],
                   None,
                   meta_train=False, mark=0)
            mmd_loss = mmd_loss.mean()
            cls_loss = cls_loss.mean()
            gamma = 2 / (1 + math.exp(-10 * (i + iters) / (iteration * epochs))) - 1 # 0.2
            loss = gamma * mmd_loss + cls_loss
            loss.backward()
            optimizer.step()
            if log:
                writer.add_scalar(f"Pretrain/{source_name}_loss", loss.item(), i+iters)
                writer.add_scalar(f"Pretrain/{source_name}_cls_loss", cls_loss.item(), i+iters)
                writer.add_scalar(f"Pretrain/{source_name}_mmd_loss", mmd_loss.item(), i+iters)
                writer.add_scalar('Pretrain/lr', optimizer.param_groups[0]['lr'], i+iters)
                writer.flush()
        
        iters += iteration
        src_acc = test_acc(model, source1_test_loader, source_name, source_type="Source")
        tgt_acc = test_acc(model, target_test_loader, target_name, source_type="Target")
        if log:
            writer.add_scalar(f"Pretrain/{source_name}_acc", src_acc, ep)
            writer.add_scalar(f"Pretrain/{target_name}_acc", tgt_acc, ep)
            writer.flush()
        
    return model, optimizer, lr_scheduler

def save_model(ckpt_path, model, config, optimizer, lr_scheduler, epoch, name="epoch-last.pth"):
    if config.get('_parallel'):
        model_ = model.module
    else:
        model_ = model
    
    training = {
      'optimizer': config['optimizer'],
      'optimizer_args': config['optimizer_args'],
      'optimizer_state_dict': optimizer.state_dict(),
      'lr_scheduler_state_dict': lr_scheduler.state_dict() 
        if lr_scheduler is not None else None,
      'epoch': epoch
    }
    
    ckpt = {
      'file': __file__,
      'config': config,

      'encoder': config['encoder'],
      'encoder_args': config['encoder_args'],
      'encoder_state_dict': model_.encoder.state_dict(),
      
      'add': config['add'],
      'add_state_dict': model_.add.state_dict(),
      'add_args': config['add_args'],
        
      'classifier': config['classifier'],
      'classifier_args': config['classifier_args'],
      'classifier_state_dict': model_.classifier.state_dict(),

      'training': training,
    }
    
    torch.save(ckpt, os.path.join(ckpt_path, name))

def load_model(config, load_path, inner_args, keep_on=True, name="epoch-last.pth"):
    """
    keep_on：是否为继续训练
    """
    load_path = os.path.join(load_path, name)
    ckpt = torch.load(load_path)
    config['encoder'] = ckpt['encoder']
    config['encoder_args'] = ckpt['encoder_args']
    config['add'] = ckpt['add']
    config['add_args'] = ckpt['add_args']
    config['classifier'] = ckpt['classifier']
    config['classifier_args'] = ckpt['classifier_args']
    model = models.load(ckpt,
                        load_clf=(not inner_args['reset_classifier']))
    optimizer, lr_scheduler = optimizers.load(ckpt, [{'params': model.encoder.parameters()},
                                             {'params': model.add.parameters()},
                                             {'params': model.classifier.parameters()}])
    if keep_on:
        start_epoch = ckpt['training']['epoch'] + 1
    else:
        start_epoch = 0
    return model, optimizer, lr_scheduler, start_epoch
    
def train(config):
    inner_args = utils.config_inner_args(config.get('inner_args'))
    if config.get('load'):
        model, optimizer, lr_scheduler, start_epoch = load_model(config, config['load'], inner_args, keep_on=True)
    else:
        config['encoder_args'] = config.get('encoder_args') or dict()
        config['add_args'] = config.get('add_args') or dict()
        config['classifier_args'] = config.get('classifier_args') or dict()
        config['encoder_args']['bn_args']['n_episode'] = config['train'][
            'n_episode']
        config['classifier_args']['n_way'] = config['train']['n_way']
        model = models.make(config['encoder'], config['encoder_args'],
                            config['classifier'], config['classifier_args'], config['add'], config['add_args'])
        optimizer, lr_scheduler = optimizers.make(config['optimizer'],
                                            [{'params': model.encoder.parameters()},
                                             {'params': model.add.parameters()},
                                             {'params': model.classifier.parameters()}],
                                            **config['optimizer_args'])
        start_epoch = 1

    ckpt_name = config['encoder']
    ckpt_name += '_' + config['dataset'].replace('meta-', '')
    
    if args.mode == 1:
        exp0 = 'w-ml'
    else:
        exp0 = 'wo-ml'
    exp = exp0 + "-" + str(args.seed) + "-" + args.s1 + "-" + args.s2 + "-Rep" + "-wGate" + "-Final"
    exp1 = exp0 + "-" + str(args.seed) + "-" + args.s1 + "-" + args.s2 + "-Rep" + "-wGate" + "-Final" + str(args.threshold) + "-ol" + str(args.on_lbd) + "ps" + str(args.push_scale)
    ckpt_path = os.path.join('./save', ckpt_name)
    writer = SummaryWriter(os.path.join(ckpt_path, exp1))
        
    if config.get('efficient'):
        model.go_efficient()

    if config.get('_parallel'):
        model = nn.DataParallel(model)
    
    if config['re_pretrain']:
        model, optimizer, lr_scheduler = pretrain(model, optimizer, sources[0], original_target_name, lr_scheduler, epochs=config['pre_train'], log=True, writer=writer)
        save_model(ckpt_path, model, config, optimizer, lr_scheduler, epoch=(config['pre_train']-1), name=f"{exp}.pth")
    else:
        model, optimizer, lr_scheduler, start_epoch = load_model(config, config['load_pretrain'], inner_args, keep_on=False, name=f"{exp}.pth")
        
    if cuda:
        model.cuda()
        
    if config.get('efficient'):
        model.go_efficient()

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    ######## Incremental Learning
    batch_size =64
    outer_flags=[3]
    inner_flags=[6]
    for outer_flag in outer_flags:
        for inner_flag in inner_flags:
        #for inner_flag in range(1,7):
            exp_name = f'o{outer_flag}_i{inner_flag}'

            model, optimizer, lr_scheduler, start_epoch = load_model(config, config['load_pretrain'], inner_args, keep_on=False, name=f"{exp}.pth")

            if cuda:
                model.cuda()

            if config.get('efficient'):
                model.go_efficient()

            if config.get('_parallel'):
                model = nn.DataParallel(model)
            
            tgt_qry_idx = []
            tgt_qry_label = [] 
            for original_source_name in sources:
                if dataset == "office31":
                    source_name = original_source_name + "/images/"
                    target_name = original_target_name + "/images/"
                else:
                    source_name, target_name = original_source_name, original_target_name
                source1_test_loader = data_loader.load_testing(root_path, source_name, batch_size, kwargs)
                target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)
                correct = 0

                epochs = config['fine_tune']
                # fine-tune
                for epoch in range(start_epoch, config['fine_tune']):
                    print(f"{epoch}/{config['fine_tune']}")
                    model.train()

                    index = hnswlib.Index(space='l2', dim=config['hnsw']['dim'])
                    index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
                    index.set_ef(int(k * 1.2))

                    start = time.time()
                    source_nums, src_qry_label, acc_src = test_hnsw(model, source1_test_loader, original_source_name, loader_type=1, index=index)
                    tgt_spt_idx, tgt_qry_idx, tgt_spt_src, tgt_spt_src_prob, tgt_qry_label, acc_tgt, tgt_index = split_target(model, target_test_loader, original_target_name, loader_type=0, idx_init=source_nums, k=10, index=index, last_simple_idx=tgt_qry_idx, last_simple_label=tgt_qry_label)
                    sim_tgt_idx, sim_tgt_prob = test_hnsw(model, source1_test_loader, original_source_name, loader_type=2, k=10, index=tgt_index)
                    end = time.time()
                    print("HNSW:", end - start)

                    cls_source1_loader = data_loader.load_training(root_path, source_name, batch_size // 4, kwargs)
                    cls_target_loader = data_loader.load_training(root_path, target_name, batch_size // 4, kwargs)

                    tgt_qry_tst_idx = np.array(tgt_qry_idx, dtype=np.int64).flatten()
                    tgt_tst_simple_loader = data_loader.load_testing_index(root_path, target_name, batch_size//2, tgt_qry_tst_idx, kwargs)
                    iters = 0
                    iteration = len(cls_source1_loader)
                    
                    # meta-testing dataloader
                    source_loader_index = np.array(range(len(cls_source1_loader.dataset))) # 源域索引
                    src_idx, tgt_idx = index_generate_sim(source_loader_index, sim_tgt_idx, sim_tgt_prob)
                    tst_src_pair1_sim_loader = data_loader.load_training_index(root_path, source_name, batch_size//2, src_idx, kwargs)
                    tst_tgt_pair1_sim_loader = data_loader.load_training_index(root_path, target_name, batch_size//2, tgt_idx, kwargs)

                    tgt_qry_idx1, tgt_qry_label1, tgt_qry_idx2, tgt_qry_label2 = index_generate_diff(tgt_qry_idx, tgt_qry_label, shuffle=True)
                    tst_tgt_pair2_diff_loader1 = data_loader.load_training_index(root_path, target_name, batch_size//2, tgt_qry_idx1, kwargs, target=True, pseudo=tgt_qry_label1)
                    tst_tgt_pair2_diff_loader2 = data_loader.load_training_index(root_path, target_name, batch_size//2, tgt_qry_idx2, kwargs, target=True, pseudo=tgt_qry_label2)    

                    pair1_iter_a = iter(tst_src_pair1_sim_loader)
                    pair1_iter_b = iter(tst_tgt_pair1_sim_loader) 

                    pair2_iter_a = iter(tst_tgt_pair2_diff_loader1)
                    pair2_iter_b = iter(tst_tgt_pair2_diff_loader2)

                    # meta-testing dataloader
                    if len(tgt_spt_idx) > batch_size//2:
                        tgt_idx, src_indices = index_generate_sim(tgt_idx=tgt_spt_idx, src_idx=tgt_spt_src, src_prob=tgt_spt_src_prob)
                        trn_tgt_pair3_sim_loader1 = data_loader.load_training_index(root_path, target_name, batch_size//2, tgt_idx, kwargs)
                        trn_src_pair3_sim_loader2 = data_loader.load_training_index(root_path, source_name, batch_size //2, src_indices, kwargs)
                    else:
                        trn_tgt_pair3_sim_loader1 = None
                        trn_src_pair3_sim_loader2 = None

                    src_qry_idx1, _, src_qry_idx2, _ = index_generate_diff(src_indices, src_qry_label[src_indices], shuffle=False)
                    trn_src_pair4_diff_loader = data_loader.load_training_index(root_path, source_name, batch_size//2, src_qry_idx2, kwargs)
                    
                    if trn_tgt_pair3_sim_loader1 is not None:
                        pair3_iter_a = iter(trn_tgt_pair3_sim_loader1)
                        pair3_iter_b = iter(trn_src_pair3_sim_loader2)
                    else:
                        pair3_iter_a = None
                        pair3_iter_b = None

                    pair4_iter_b = iter(trn_src_pair4_diff_loader)

                    trn_cls = iter(cls_source1_loader)
                    trn_cls_tgt = iter(cls_target_loader)

                    ############### Meta-Trainging
                    it = 0
                    optimizer.param_groups[0]['lr'] = lr[0] / (2 *math.pow((1 + 10* iters / (iteration * epochs)), 0.7))
                    optimizer.param_groups[1]['lr'] = lr[1] / (2 *math.pow((1 + 10* iters / (iteration  * epochs )), 0.7))
                    optimizer.param_groups[2]['lr'] = lr[1] / (2 *math.pow((1 + 10* iters / (iteration  * epochs)), 0.7))
                    inner_args['encoder_lr']=optimizer.param_groups[0]['lr'] / lr_scale[1]
                    inner_args['add_lr']=optimizer.param_groups[1]['lr'] / lr_scale[1]
                    inner_args['classifier_lr']=optimizer.param_groups[1]['lr'] / lr_scale[1]
                    
                    training_process = tqdm(cls_source1_loader, desc='fine-tune', leave=False)
                    for data in training_process:
                        # similar samples of the simple source samples
                        pair1_src_data, pair1_src_label, pair1_iter_a = save_iter(pair1_iter_a, tst_src_pair1_sim_loader)
                        pair1_tgt_data, pair1_tgt_label, pair1_iter_b = save_iter(pair1_iter_b, tst_tgt_pair1_sim_loader)
                        trn_pair1 = [pair1_src_data, pair1_src_label, pair1_tgt_data, pair1_tgt_label]
                        # same domain & different classes
                        pair2_tgt_data1, pair2_tgt_label1, pair2_iter_a = save_iter(pair2_iter_a, tst_tgt_pair2_diff_loader1)
                        pair2_tgt_data2, pair2_tgt_label2, pair2_iter_b = save_iter(pair2_iter_b, tst_tgt_pair2_diff_loader2)
                        trn_pair2 = [pair2_tgt_data1, pair2_tgt_label1, pair2_tgt_data2, pair2_tgt_label2]

                        trn_data = [trn_pair1, trn_pair2]
                        # similar samples of the hard target samples
                        if pair3_iter_a is not None:
                            pair1_tgt_data, pair1_tgt_label,  pair3_iter_a = save_iter(pair3_iter_a, trn_tgt_pair3_sim_loader1)
                            pair1_src_data, pair1_src_label,  pair3_iter_b = save_iter(pair3_iter_b, trn_src_pair3_sim_loader2)
                            tst_pair1 = [pair1_tgt_data, pair1_tgt_label, pair1_src_data, pair1_src_label]
                        else:
                            tst_pair1 = None
                        # (source domain) same domain & different class
                        pair2_src_data2, pair2_src_label2, pair4_iter_b = save_iter(pair4_iter_b, trn_src_pair4_diff_loader)
                        tst_pair2 = [pair2_src_data2, pair2_src_label2]
                        # cross entropy & mmd
                        tgt_trn_data, tgt_trn_label, trn_cls_tgt = save_iter(trn_cls_tgt, cls_target_loader)
                        src_trn_data, src_trn_label = data
                        tst_group = [src_trn_data, src_trn_label, tgt_trn_data, tgt_trn_label]

                        tst_data = [tst_pair1, tst_pair2, tst_group]
                        
                        
                        gamma = 2 / (1 + math.exp(-10 * (it) / (iteration * epochs) )) - 1
                        optimizer.zero_grad()
                        loss, pull_loss, push_loss = model(trn_data, tst_data,inner_args,meta_train=True, mark=2,trn_flag=inner_flag, on_lbd=args.on_lbd, scale=args.push_scale, mmd_alpha=gamma)

                        pull_loss = pull_loss.mean()
                        push_loss = push_loss.mean()
                        loss =loss.mean()
                        training_process.set_postfix(pull=pull_loss.item())
                        training_process.set_postfix(push=push_loss.item())
                        loss.backward()
                        writer.add_scalar(f"loss/finetune_loss_{original_source_name}_{exp_name}", loss.item(), iters)
                        writer.add_scalar(f"loss/pull_loss_{original_source_name}_{exp_name}", pull_loss.item(), iters)
                        writer.add_scalar(f"loss/push_loss_{original_source_name}_{exp_name}", push_loss.item(), iters)
                        writer.add_scalar(f'parameters/finetune_lr_{original_source_name}_{exp_name}', optimizer.param_groups[0]['lr'], iters)

                        optimizer.step()
                        writer.flush()
                        iters += len(data)
                        it += len(data)

                    src_acc = test_acc(model, source1_test_loader, source_name, source_type="Source")
                    tgt_acc = test_acc(model, target_test_loader, target_name, source_type="Target")
                    tgt_sim_acc = test_acc(model, tgt_tst_simple_loader, target_name, source_type="Target", sampler=True)
                    tgt_hard_acc = (len(target_test_loader.dataset) * tgt_acc - len(tgt_tst_simple_loader.sampler) * tgt_sim_acc) * 100.0 / (len(target_test_loader.dataset) - len(tgt_tst_simple_loader.sampler))
                    writer.add_scalar(f"Acc/source_{original_source_name}_{exp_name}", src_acc, epoch + config['meta_epochs'])
                    writer.add_scalar(f"Acc/target_{original_target_name}_{original_source_name}_{exp_name}", tgt_acc, epoch + config['meta_epochs'])
                    writer.add_scalar(f"Acc/target_simple_{original_target_name}_{original_source_name}_{exp_name}", tgt_sim_acc, epoch + config['meta_epochs'])
                    writer.add_scalar(f"Acc/target_hard_{original_target_name}_{original_source_name}_{exp_name}", tgt_hard_acc, epoch + config['meta_epochs'])
                    writer.add_scalar(f"Acc/target_simple_num_{original_target_name}_{original_source_name}_{exp_name}", len(tgt_tst_simple_loader.sampler), epoch + config['meta_epochs'])
                    writer.flush()
                    
                save_model(ckpt_path, model, config, optimizer, lr_scheduler, epoch=(config['fine_tune'] + config['meta_epochs']), name=f"{original_source_name}_{exp}.pth")

                
def test_acc(model, data_loader, source_name, source_type="source", sampler=False):
    """
    loader_type: 1 source domain，0 target domain
    """
    test_loss = 0
    correct = 0
    num=0
    with torch.no_grad():
        for data, target in data_loader:
            padding = 0
            if data.shape[0] < batch_size:
                padding = batch_size - data.shape[0]
                zeros_shape = data.shape
                zeros = torch.zeros((padding, zeros_shape[1], zeros_shape[2], zeros_shape[3]))
                data = torch.cat([data, zeros])
                zeros_tgt = torch.zeros((padding, ))
                target = torch.cat([target, zeros_tgt])
                
            data, target = Variable(data), Variable(target)
            data, target = data.cuda(), target.cuda()
            pred, _ = model(None,
                   [data, target],
                   None,
                   meta_train=False, mark=3)

            if padding > 0:
                pred = pred[:(data.shape[0] - padding)]
                target = target[:(data.shape[0] - padding)]
            
            pred = torch.nn.functional.softmax(pred, dim=1)
            pred = pred.data.max(1)[1].cpu()
            correct += pred.eq(target.cpu().data.view_as(pred)).cpu().sum()
            num+=data.shape[0]
            
            
        if sampler:
            acc = 100. * correct / len(data_loader.sampler)
            print(source_name, 'Accuracy: {}/{} ({:.4f}%)\n'.format(correct, len(data_loader.sampler), acc))
        else:
            acc = 100. * correct / len(data_loader.dataset)
            print(source_name, 'Accuracy: {}/{} ({:.4f}%)\n'.format(correct,  len(data_loader.dataset), acc))
    return acc    

def split_target(model, data_loader, source_name, loader_type=1, idx_init=0, k=10, index=None, last_simple_idx=None, last_simple_label=None):
    """
    loader_type: 1 source domain，0 target domain
    last_simple_idx: the total simple samples before last epoch;
    last_simple_label: the labels;
    """
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    tgt_spt_src = np.empty(shape=(0, k))
    tgt_spt_src_prob = np.empty(shape=(0, k), dtype=np.float32)
    tgt_spt_idx = []
    tgt_qry_idx = []
    tgt_qry_label = []
    src_qry_label = np.empty(shape=(0))
    sim_tgt_idx = np.empty(shape=(0, k))
    sim_tgt_prob = np.empty(shape=(0, k), dtype=np.float32)
    idx = idx_init
    domain_idx = 0

    # build HNSW
    tgt_index = hnswlib.Index(space='l2', dim=config['hnsw']['dim']) # 
    tgt_index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
    tgt_index.set_ef(int(k * 1.2))
    
    # 
    tgt_logit = np.empty(shape=(0, config['train']['n_way']), dtype=np.float32)
    tgt_feat = np.empty(shape=(0, 256), dtype=np.float32)
    
    with torch.no_grad():
        for data, target in data_loader:
            padding = 0
            if data.shape[0] < batch_size:
                padding = batch_size - data.shape[0]
                zeros_shape = data.shape
                zeros = torch.zeros((padding, zeros_shape[1], zeros_shape[2], zeros_shape[3]))
                data = torch.cat([data, zeros])
                zeros_tgt = torch.zeros((padding, ))
                target = torch.cat([target, zeros_tgt])
                
            data, target = Variable(data), Variable(target)
            data, target = data.cuda(), target.cuda()

            logit, feat = model(None,
                   [data, target],
                   None,
                   meta_train=False, mark=3)
            if padding > 0:
                logit = logit[:(data.shape[0] - padding)]
                target = target[:(data.shape[0] - padding)]
                feat = feat[:(data.shape[0] - padding)]
            
            pred = torch.nn.functional.softmax(logit, dim=1)

            feat = feat.cpu().numpy()
            
            tgt_logit = np.concatenate((tgt_logit, logit.cpu().numpy()), axis=0)
            tgt_feat = np.concatenate((tgt_feat, feat), axis=0)
            
            pred = pred.data.max(1)[1].cpu()
            correct += pred.eq(target.cpu().data.view_as(pred)).cpu().sum()

            idx += len(feat)
            domain_idx += len(feat)
            
    output = softmax(tgt_logit)
    simple_relative = np.argwhere(output >= hard_threshold)
    idx, pseudo = zip(*simple_relative)
    simple_idx = []
    simple_label = []
    hard_idx = []
    
    for i in range(config['train']['n_way']):
        query_idx = np.argwhere(np.array(pseudo)==i).flatten()
        if len(query_idx) > 0:
            sub_samples = simple_relative[query_idx]
            sub_idx, _ = zip(*sub_samples)
            sub_idx = np.array(sub_idx)
            sub_feat = tgt_feat[sub_idx]
            sub_dist = np.linalg.norm(sub_feat - np.mean(sub_feat, axis=0), ord=2, axis=1)
            sub_dist_ord = sub_dist.argsort()
            cut_len = len(sub_idx) * 9 // 10

            simple_label.extend([i] * cut_len)
            simple_idx.extend(sub_idx[sub_dist_ord[:cut_len]])
            hard_idx.extend(sub_idx[sub_dist_ord[cut_len:]])
    
    for i, x in enumerate(simple_idx):
        if x not in last_simple_idx:
            last_simple_idx.append(x)
            last_simple_label.append(simple_label[i])
            
    
    hard_idx = [x for x in range(len(data_loader.dataset)) if x not in last_simple_idx]
    
    for x in simple_idx:
        tgt_index.add_items(tgt_feat[x], x)

    for x in hard_idx:
        labels, distances = index.knn_query(tgt_feat[x], k=k)
        tgt_spt_src = np.concatenate((tgt_spt_src, labels), axis=0)
        tgt_spt_src_prob = np.concatenate((tgt_spt_src_prob, softmax(distances)), axis=0)
        
    acc = 100. * correct / len(data_loader.dataset)

    return hard_idx, last_simple_idx, tgt_spt_src, tgt_spt_src_prob, last_simple_label, acc, tgt_index


def test_hnsw(model, data_loader, source_name, loader_type=1, idx_init=0, k=10, index=None):
    """
    loader_type: 1 source domain，0 target domain
    """
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    tgt_spt_src = np.empty(shape=(0, k))
    tgt_spt_src_prob = np.empty(shape=(0, k), dtype=np.float32)
    tgt_spt_idx = []
    tgt_qry_idx = []
    tgt_qry_label = []
    src_qry_label = np.empty(shape=(0))
    sim_tgt_idx = np.empty(shape=(0, k))
    sim_tgt_prob = np.empty(shape=(0, k), dtype=np.float32)
    idx = idx_init
    domain_idx = 0
    if loader_type == 0:
        tgt_index = hnswlib.Index(space='l2', dim=config['hnsw']['dim'])
        tgt_index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
        tgt_index.set_ef(int(k * 1.2))
    with torch.no_grad():
        for data, target in data_loader:
            padding = 0
            if data.shape[0] < batch_size:
                padding = batch_size - data.shape[0]
                zeros_shape = data.shape
                zeros = torch.zeros((padding, zeros_shape[1], zeros_shape[2], zeros_shape[3]))
                data = torch.cat([data, zeros])
                zeros_tgt = torch.zeros((padding, ))
                target = torch.cat([target, zeros_tgt])
                
            data, target = Variable(data), Variable(target)
            data, target = data.cuda(), target.cuda()

            pred, feat = model(None,
                   [data, target],
                   None,
                   meta_train=False, mark=3)
            if padding > 0:
                pred = pred[:(data.shape[0] - padding)]
                target = target[:(data.shape[0] - padding)]
                feat = feat[:(data.shape[0] - padding)]
            
            pred = torch.nn.functional.softmax(pred, dim=1)

            feat = feat.cpu().numpy()
            if loader_type == 0:
                tgt_hard = torch.where(pred.data.max(1)[0] < hard_threshold)[0].cpu().numpy()
                tgt_simple = torch.where(pred.data.max(1)[0] >= hard_threshold)[0].cpu().numpy()
                tgt_spt_idx = np.concatenate((tgt_spt_idx, [x + domain_idx for x in tgt_hard]), axis=0)
                tgt_qry_idx = np.concatenate((tgt_qry_idx, [x + domain_idx for x in tgt_simple]), axis=0)
                tgt_pred = pred.data.max(1)[1].cpu().numpy()
                tgt_qry_label = np.concatenate((tgt_qry_label, tgt_pred[tgt_simple]), axis=0)
                
                tgt_index.add_items(feat[tgt_simple], [x + domain_idx for x in tgt_simple])
                
                for spt_sample in tgt_hard:
                    labels, distances = index.knn_query(feat[spt_sample], k=k)
                    tgt_spt_src = np.concatenate((tgt_spt_src, labels), axis=0)
                    tgt_spt_src_prob = np.concatenate((tgt_spt_src_prob, softmax(distances)), axis=0)

            elif loader_type == 1:
                index.add_items(feat, [x + domain_idx for x in range(len(feat))])
                src_qry_label = np.concatenate([src_qry_label, target.cpu().numpy()], axis=0)
            elif loader_type == 2:
                for i in range(len(feat)):
                    labels, distances = index.knn_query(feat[i], k=k)
                    sim_tgt_idx = np.concatenate((sim_tgt_idx, labels), axis=0)
                    sim_tgt_prob = np.concatenate((sim_tgt_prob, softmax(distances)), axis=0)
                    
            pred = pred.data.max(1)[1].cpu()
            correct += pred.eq(target.cpu().data.view_as(pred)).cpu().sum()

            idx += len(feat)
            domain_idx += len(feat)
        
        acc = 100. * correct / len(data_loader.dataset)
        print(source_name, 'Accuracy: {}/{} ({:.4f}%)\n'.format(
         correct, len(data_loader.dataset), acc))
    if loader_type == 1:
        return idx, src_qry_label, acc
    elif loader_type == 0:
        return tgt_spt_idx, tgt_qry_idx, tgt_spt_src, tgt_spt_src_prob, tgt_qry_label, acc, tgt_index
    else:
        return sim_tgt_idx, sim_tgt_prob

def save_iter(loader_iterator, dataloader):
    try:
        data, label = loader_iterator.next()
    except:
        loader_iterator = iter(dataloader)
        data, label = loader_iterator.next()
    return data, label, loader_iterator
        

def index_generate_sim(tgt_idx, src_idx, src_prob):
    src_indices = [np.random.choice(x, size=1, p=src_prob[i]) for i,x in enumerate(src_idx)]
    idx_pair = list(zip(tgt_idx, src_indices))
    np.random.shuffle(idx_pair)
    tgt_idx, src_indices = zip(*idx_pair)
    tgt_idx = np.array(tgt_idx, dtype=np.int64).flatten()
    src_indices = np.array(src_indices, dtype=np.int64).flatten()
    return tgt_idx, src_indices


def index_generate_diff(tgt_qry_idx, tgt_qry_label, shuffle):
    """
    """
    d = defaultdict(list)
    for s, l in zip(tgt_qry_idx, tgt_qry_label):
        d[l].append(s)
    id2label = dict(zip(tgt_qry_idx, tgt_qry_label))
    category_list = list(d.keys())
    tgt_qry_idx_aux = [np.random.choice(d[random_chice_the_other(tgt_qry_label[i],category_list)], size=1, replace=True) for i in range(len(tgt_qry_idx))]
    if shuffle:
        idx_pair = list(zip(tgt_qry_idx, tgt_qry_idx_aux))
        np.random.shuffle(idx_pair)
        tgt_qry_idx, tgt_qry_idx_aux = zip(*idx_pair)
    tgt_qry_idx = np.array(tgt_qry_idx, dtype=np.int64).flatten()
    tgt_qry_idx_aux = np.array(tgt_qry_idx_aux, dtype=np.int64).flatten()

    tgt_qry_label = np.array([id2label[x] for x in tgt_qry_idx], dtype=np.int64)
    tgt_qry_label_aux = np.array([id2label[x] for x in tgt_qry_idx_aux], dtype=np.int64)

    return tgt_qry_idx, tgt_qry_label, tgt_qry_idx_aux, tgt_qry_label_aux
    
def random_chice_the_other(x, categorys):
    categorys = set(categorys)
    categorys.discard(x)
    return int(np.random.choice(list(categorys), size=1).item())
    

    
def softmax(x):
    """ softmax function """
    
    x -= np.max(x, axis = 1, keepdims = True)
    
    x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
    
    return x


if __name__ == '__main__':
    if args.mode == 1:
        config = yaml.load(open("configs/wml.yaml", 'r'), Loader=yaml.FullLoader)
    else:
        config = yaml.load(open("configs/woml.yaml", 'r'), Loader=yaml.FullLoader)
    train(config)