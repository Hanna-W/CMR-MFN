import torch
import torch.nn as nn
from model_T import TSN
import copy
from torch.nn.init import normal_, constant_
import torch.nn.functional as F
import logging

class Attention(nn.Module):
    def __init__(self,
                 dim,  # input dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.5,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, 128)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x,attn


class Fusion_Network(nn.Module):

    def __init__(self, input_dim, modality, fusion_type, dropout, num_segments):
        super().__init__()
        self.input_dim = input_dim
        self.num_segments=num_segments
        self.modality = modality
        self.fusion_type = fusion_type
        self.dropout = dropout

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)
        self.selfat = Attention(self.input_dim)

    def forward(self, inputs):
        outs = []
        if len(self.modality) > 1:  # Multi modality: Fusion
            if self.fusion_type == 'attention':
                for m in self.modality:
                    out = inputs[m]
                    out = out.unsqueeze(1)
                    outs.append(out)
                out = torch.cat(outs, dim=1)
                out = self.selfat(out)
                base_out = torch.mean(out[0],1)

        else:  # Single modality
            base_out = inputs['RGB']


        output = {'features': base_out}

        return output
    
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

        return self


class Classification_Network(nn.Module):
    def __init__(self, feature_dim, modality, num_class,
                 dropout, num_segments):
        super().__init__()
        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.dropout = dropout

        self._add_classification_layer(feature_dim)

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)

    def _add_classification_layer(self, input_dim):

        std = 0.001

        self.fc_action = nn.Linear(input_dim, self.num_class)
        normal_(self.fc_action.weight, 0, std)
        constant_(self.fc_action.bias, 0)
        self.weight = self.fc_action.weight
        self.bias = self.fc_action.bias

    def forward(self, inputs):
            
        base_out = self.dropout_layer(inputs)
        base_out = self.fc_action(base_out)
        output = {'logits': base_out}
        return output


class CMR_MFNNet(nn.Module):
    def __init__(self, num_segments, modality, base_model='ViT',
                 new_length=None, dropout=0.5, fusion_type='concat'):
        super().__init__()

        self.fusion_networks = nn.ModuleList()
        self.fc_list = nn.ModuleList()
        self.fc = None

        self.num_segments = num_segments
        self.modality = modality
        self.base_model = base_model
        self.new_length = new_length
        self.dropout = dropout
        self.fusion_type = fusion_type

        self.feature_extract_network = TSN(self.num_segments, self.modality,
                                           self.base_model, self.new_length)

        self.fusion_network = Fusion_Network(768, self.modality, self.fusion_type, self.dropout, self.num_segments)

        logging.info((
"""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    dropout_ratio:      {}
""".format(base_model, self.modality, self.num_segments, self.feature_extract_network.new_length,
self.dropout)))

    @property
    def feature_dim(self):
        if len(self.modality) > 1:
            if self.fusion_type == 'attention':
                return 128
            elif self.fusion_type == 'concat':
                return 768*len(self.modality)
        else:
            return 768

    def get_convnet(self, convnet_type):
        name = convnet_type.lower()
        if name == "fusion":
            model = Fusion_Network(768, self.modality, self.fusion_type, self.dropout, self.num_segments)
            return model
        elif name == 'fc':
            in_features = self.fc.fc_action.in_features
            num_classes = self.fc.num_class
            model = Classification_Network(in_features, self.modality, num_classes,
                                           self.dropout, self.num_segments)
            return model
        else:
            raise NotImplementedError("Unknown type {}".format(convnet_type))

    def extract_vector(self, x):
        vit_features = self.feature_extract_network(x)
        fusion_feature = self.fusion_network(vit_features)["features"]
        return fusion_feature
    

    def forward(self, x, cur_task_size, mode='train'):
        vit_features = self.feature_extract_network(x)

        if mode == 'train':
            fusion_feature = self.fusion_network(vit_features)["features"]
            out = self.fc(fusion_feature)
            out.update({"features": vit_features, "fusion_features": fusion_feature})
        elif mode == 'test':
            fusion_features, logits = [], []
            for idx, fc in enumerate(self.fc_list):
                fusion_feature = self.fusion_networks[idx](vit_features)["features"]
                out = fc(fusion_feature)["logits"][:, :cur_task_size]
                fusion_features.append(fusion_feature)
                logits.append(out)

            fusion_features = torch.cat(fusion_features, 1)
            logits = torch.cat(logits, 1)
            out = {"logits": logits}
            out.update({"features": vit_features, "fusion_features": fusion_features})

        return out
        """
        {
            'features': vit_features
            'fusion_features': fusion_features
            'logits': logits
        }
        """

    def save_parameter(self):
        self.fusion_networks.append(self.get_convnet('Fusion'))
        self.fusion_networks[-1].load_state_dict(self.fusion_network.state_dict())

        if self.fc is not None:
            self.fc_list.append(self.get_convnet('FC'))
            self.fc_list[-1].load_state_dict(self.fc.state_dict())


    def _gen_train_fc(self, incre_classes):

        fc = Classification_Network(self.feature_dim, self.modality, incre_classes,
                                    self.dropout, self.num_segments)

        if self.fc is not None:
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data = weight[:incre_classes]
            fc.bias.data = bias[:incre_classes]

        del self.fc
        self.fc = fc
        
        # fc = Classification_Network(self.feature_dim, self.modality, incre_classes,
        #                             self.dropout, self.before_softmax, self.num_segments)
        # del self.fc
        # self.fc = fc

        # fusion_network=Fusion_Network(768, self.modality, self.midfusion, self.dropout, self.num_segments)
        # del self.fusion_network
        # self.fusion_network = fusion_network

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self