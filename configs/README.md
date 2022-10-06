dataset: meta-mini-imagenet
train:
  split: meta-train
  image_size: 84
  normalization: True
  transform: flip
  n_batch: 200
  n_episode: 50
  n_way: 31
  n_shot: 1
  n_query: 15
val:
  split: meta-val
  image_size: 84
  normalization: True
  transform: flip
  n_batch: 200
  n_episode: 50
  n_way: 5
  n_shot: 1
  n_query: 15

encoder: resnet50
encoder_args:
  bn_args:
    track_running_stats: True
add: addneck // 添加的特征
add_args:
  planes: 256

classifier: logistic

inner_args:
  reset_classifier: False
  n_step: 1
  encoder_lr: 0.001
  classifier_lr: 0.001
  momentum: 0.9
  weight_decay: 5.e-4
  first_order: False

optimizer: sgd
optimizer_args:
  lr: 0.01
  weight_decay: 5.e-4
  schedule: step
  milestones:
    - 100
    - 300
    - 450
    
load_pretrain: /root/workspace/ISDA/MUDA/MFSAN/MFSAN_2src/save/resnet50_mini-imagenet/epoch-last.pth // 加载首歌源域的预训练模型
re_pretrain: True // 首个源域时是否进行预训练，否则指定load_pretrain
_parallel: True // 是否启用多卡训练
pre_train: 50 // 首个源域的预训练轮数
meta_epochs: 20 // meta-learning轮数
fine_tune: 20 //fine-tune轮数

hnsw:
  dim: 256