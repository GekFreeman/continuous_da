from torchvision import datasets, transforms
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import pdb
import random
from collections import defaultdict

def sample_sameclass(source_data,source_label,tgt_data,tgt_label):
    index_s=[]
    index_t=[]
    b=source_data.shape[0]
    c,h,w=source_data.shape[1],source_data.shape[2],source_data.shape[3]
    tgt_index=[i for i in range(tgt_data.shape[0])]
    dic=defaultdict(list)
    for dat,lab in zip(tgt_index,tgt_label):
        #if lab==dic.keys():
        key=lab.detach().cpu().numpy()
        key=str(key)
        dic[key].append(dat)
    #t=0
    for s in range(len(source_label)):
        s_idx=str(source_label[s].detach().cpu().numpy())        
        if s_idx in dic.keys():#在源域中找到与目标域一样的类，记录编号
                t=random.choice(dic[s_idx])
                index_s.append(s)
                index_t.append(t)
    num=len(index_s)
    s_data=torch.zeros((num,c,h,w),dtype=torch.float).cuda()
    s_label=torch.zeros(num,dtype=torch.long).cuda()
    t_data=torch.zeros((num,c,h,w),dtype=torch.float).cuda()
    t_label=torch.zeros(num,dtype=torch.long).cuda()             
   # print(s_data.dtype)
    for i in range(num):#按照编号去处数据替换原来值
        index=index_s[i]
        s_data[i,:]=source_data[index,:]
        s_label[i]=source_label[index]
    for j in range(num):
        index=index_t[j]
        t_data[j,:]=tgt_data[index,:]
        t_label[j]=tgt_label[index]
    return s_data, s_label ,t_data,t_label