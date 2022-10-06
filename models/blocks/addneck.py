from collections import OrderedDict
import pdb
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .blocks import register
from ..modules import *

__all__ = ['addneck']


def conv3x3(in_channels, out_channels, stride=1):
    return Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)

def conv7x7(in_channels, out_channels, stride=1):
    return Conv2d(in_channels, out_channels, 7, stride, padding=3, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    return Conv2d(in_channels, out_channels, 1, stride, padding=0, bias=False)


class ADDneck(Module):

    def __init__(self, inplanes, planes, bn_args, stride=1, downsample=None):
        super(ADDneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2d(planes, **bn_args)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = BatchNorm2d(planes, **bn_args)
        self.conv3 = conv1x1(planes, planes)
        self.bn3 = BatchNorm2d(planes, **bn_args)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.stride = stride
        self.planes = planes

    def forward(self, x, params=None, episode=None):

        out = self.conv1(x, get_child_dict(params, 'conv1'))
        out = self.bn1(out, get_child_dict(params, 'bn1'), episode)
        out = self.relu(out)

        out = self.conv2(out, get_child_dict(params, 'conv2'))
        out = self.bn2(out, get_child_dict(params, 'bn2'), episode)
        out = self.relu(out)

        out = self.conv3(out, get_child_dict(params, 'conv3'))
        out = self.bn3(out, get_child_dict(params, 'bn3'), episode)
        out = self.relu(out)
      #  out = self.avgpool(out)
      #  out = out.view(out.size(0),-1)
        return out
    
    def get_out_dim(self):
        return self.planes
    
@register('addneck')
def addneck(inplanes, planes, bn_args=dict()):
    model = ADDneck(inplanes, planes, bn_args)
    return model