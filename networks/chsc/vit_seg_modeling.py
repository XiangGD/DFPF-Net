# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm, BatchNorm2d
from torch.nn.modules.utils import _pair
from scipy import ndimage
import torch.nn.functional as F
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
#测试模型
#import vit_seg_configs as configs
#from vit_seg_modeling_resnet_skip import ResNetV2
#from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)
    

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Channel_Conv2d(nn.Module):
    def __init__(self ):        
        super(Channel_Conv2d,self).__init__()
    def forward(self,x,kernel):
        return F.conv2d(x,kernel, stride=1, padding=2)
    
class Constraint_Conv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=5, stride=1, padding=2):
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(3, 1, 1) * -1.000)
        super(Constraint_Conv2d, self).__init__()
        
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        #self.kernel = nn.Parameter(torch.rand(3, 1, kernel_size ** 2 - 1), requires_grad=True ) 
        self.kernel =torch.rand(3, 1, kernel_size ** 2 - 1)    
        
        self.kernel_r = nn.Parameter(torch.cat((self.bayarConstraint(), self.highpassConstraint()),dim=0), requires_grad=True)
        self.kernel_g = nn.Parameter(torch.cat((self.bayarConstraint(), self.highpassConstraint()),dim=0), requires_grad=True)
        self.kernel_b = nn.Parameter(torch.cat((self.bayarConstraint(), self.highpassConstraint()),dim=0), requires_grad=True)
        self.conv_r = Channel_Conv2d()
        self.conv_g = Channel_Conv2d()
        self.conv_b = Channel_Conv2d()
        print('kernel_r:',self.kernel_r,self.kernel_r.shape)
   
    def bayarConstraint(self):#[3,1,5,5] 
        self.kernel.data = self.kernel.permute(2, 0, 1)
        self.kernel.data = torch.div(self.kernel.data, self.kernel.data.sum(0))
        self.kernel.data = self.kernel.permute(1, 2, 0)
       
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)
        real_kernel = real_kernel.reshape((3, 1, self.kernel_size, self.kernel_size))
        print('bayar_kernel:',real_kernel.shape)
        return real_kernel
    
    def highpassConstraint(self):#[9,5,5 ]  
        hp1 = np.zeros(25).reshape(5,5).astype(np.float32)
        hp1[2:4,2:4] = np.array([[-1,0],[1,0]])    
        hp2 = np.zeros(25).reshape(5,5).astype(np.float32)
        hp2[2,2:4] = np.array([-1,1]).astype(np.float32)
        hp3 = np.zeros(25).reshape(5,5).astype(np.float32)
        hp3[2:4,2:4] = np.array([[-1,0],[0,1]])
        kernel = torch.from_numpy(np.stack([hp1,hp2,hp3],axis=0)).unsqueeze(1)#[3,1,5,5]
        print('hp:',kernel.shape)
        return kernel
      
        
    def forward(self, x):
        r,g,b = torch.chunk(x,3,dim =1)
        #print('rgb:',r.shape,g.shape,b.shape)
        x_r = self.conv_r(r,self.kernel_r)
        x_g = self.conv_g(g,self.kernel_g)
        x_b = self.conv_b(b,self.kernel_b)
        #print('xrgb:',x_r.shape,x_g.shape,x_b.shape)
        x = torch.cat((x_r,x_g,x_b),dim=1)
        #print('con_x:',x.shape)
        return x

class Dual_Features_Fusion(nn.Module):
    def __init__(self, C1_size, C2_size, C3_size, C4_size):
        super(Dual_Features_Fusion, self).__init__()

       # upsample C4 to get P3 from the FPN paper
        self.P4 = Conv2d(C4_size*2, C4_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
       
        # add P4 elementwise to C3
        self.P3_1 = Conv2d(C3_size*2, C3_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = Conv2d(C3_size*2, C3_size, kernel_size=3, stride=1, padding=1)
        self.P3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
       
        # add P3 elementwise to C2
        self.P2_1 = Conv2d(C2_size*2, C1_size*2, kernel_size=1, stride=1, padding=0)
        self.P2_2 = Conv2d(C2_size*2, C2_size, kernel_size=3, stride=1, padding=1)
        self.P2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        #add P2 elementwise to C1
        self.P1 = Conv2d(C1_size*2, C1_size, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x,n_x):
        C4, C3, C2, C1 = x
        n_C4, n_C3, n_C2, n_C1 = n_x
        
        P4_x = self.P4(torch.cat([C4,n_C4],dim=1))
        P4_upsampled_x = self.P4_upsample(P4_x)
        
        P3_x = torch.cat([C3,n_C3],dim=1)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsample(self.P3_1(P3_x))
        P3_x = self.P3_2(P3_x)
        
        P2_x = torch.cat([C2,n_C2],dim=1)
        P2_x = P2_x + P3_upsampled_x
        P2_upsampled_x = self.P2_upsample(self.P2_1(P2_x))
        P2_x  = self.P2_2(P2_x)
       
        P1_x = torch.cat([C1,n_C1],dim=1)
        P1_x = self.P1(P1_x + P2_upsampled_x)
        
        return [P3_x, P2_x, P1_x]
    
    
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""
    
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        self.fpn = Dual_Features_Fusion(64, 256, 512, 1024)
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        
        #self.feature_fusion = Conv2d(in_channels*2, in_channels, 1)    
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])
           
        assert self.hybrid_model is not None 
        self.constr_conv = Constraint_Conv2d(in_channels=3)   

    def forward(self, x):
        if self.hybrid:
            conv_x, features = self.hybrid_model(x, False)
            noise_x = self.constr_conv(x)
            noise_x, noise_features = self.hybrid_model(noise_x,True)
            #print('ResNetV2 output:',conv_x.shape,type(features),type(conv_x),type(noise_x))
        else:
            features = None
        x = self.patch_embeddings(conv_x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        #print('position_emb:',self.position_embeddings)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        features = self.fpn(features, noise_features)
        #for i in range(len(features)):
         #   print('fpn features{}:'.format(i),features[i].shape)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        #print('transformer input:',input_ids.shape)
        embedding_output, features = self.embeddings(input_ids)
        #for i in range(len(features)):
         #   print('resnet features{}:'.format(i),features[i].shape)
        #print('features len:',len(features))
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class SimilarityHead(nn.Module):
    def __init__(self, win_size):
        super().__init__()
        self.k = win_size
        self.unfold = nn.Unfold((win_size,win_size), padding=(win_size-1)//2, stride=1)
        self.sim = nn.CosineSimilarity(dim=1)
        
    def forward(self,x):
        b,c,h,w = x.shape
        #print('sim_input:',x.shape,self.k)
        x_slid = self.unfold(x)
        #print('x_slid:',x_slid.shape)
        x_slid=x_slid.reshape(b,c,self.k**2,h*w)
        x = x.reshape(b,c,h*w).unsqueeze(dim=2)
        sim_matrix = torch.mean(self.sim(x,x_slid),dim=1).reshape(b,h,w).unsqueeze(dim=1)
        #print('sim_matrix:',sim_matrix.shape)
        return sim_matrix
    
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        '''
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        '''
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.sim_head = SimilarityHead(win_size=3)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        #x = self.conv2(x)
        x = x + self.sim_head(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)
        
class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


            
class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0
        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
       

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)# （ B, hidden, n_patch)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class ForensicTransformer(nn.Module):
    def __init__(self, config, img_size=256, num_classes=1, zero_head=False, vis=False):
        super(ForensicTransformer, self).__init__()
        
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        #print(logits.shape)
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


if __name__ == '__main__':
    #img = torch.randn(2, 3, 256, 256)
    img = torch.randn( 2, 3, 256, 256)
    vit_name = 'R50-ViT-B_16'
    config_vit = CONFIGS[vit_name]
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    pretrain_path = config_vit.pretrained_path
    net1=Constraint_Conv2d(3)
    out1 = net1(img)
   
   
   # print(out1.shape)
    #net1.cuda()
    #for name,param in net1.named_parameters():   #查看可训练参数
     #  #if param.requires_grad:
      #  print('name:',name)
       # print('param:',param.size(),param.device)
    #print(next(net1.parameters()).device)
    
    
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (16,16)
    #writer = SummaryWriter('/home/user/runs')
    net = ForensicTransformer(config_vit, img_size=256, num_classes=config_vit.n_classes)
    net.load_from(weights=np.load(config_vit.pretrained_path))
    print(net)
    #writer.add_graph(net,input_to_model= img)
    net.cuda()
    img=img.cuda()
    #writer.add_graph(net,input_to_model= img)
    output = net(img)
    print(output.shape)
    #writer.close()
    #from torchinfo import summary
    #summary(net, (1,3, 256, 256))
    for name,param in net.named_parameters():   #查看可训练参数
        if param.requires_grad:
            print('name:',name)
            print('param:',param.size(),param.device)
        
    

