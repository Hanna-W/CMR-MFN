import torch
from torch import nn
from transforms import *
from collections import OrderedDict
from backbone.ViT import vit_base_patch16_224 as create_model
from TimeSformer.timesformer.models.vit import TimeSformer
import logging

class TSN(nn.Module):

    def __init__(self, num_segments, modality,
                 base_model='ViT', new_length=None):
        super(TSN, self).__init__()
        self.num_segments = num_segments
        self.base_model = base_model
        self.modality = modality

        self.new_length = OrderedDict()
        if new_length is None:
            for m in self.modality:
                if m == 'RGB':
                    self.new_length[m] = 1
                elif m == 'Accespec' or m == 'Gyrospec':
                    self.new_length[m] = 1
        else:
            self.new_length = new_length

        self._prepare_base_model(base_model)

        self._prepare_tbn()

        for m in self.modality:
            self.add_module(m.lower(), self.base_model[m])

    def _remove_classfication_layer(self):
        for m in self.modality:
            if m=="RGB":
                delattr(self.base_model[m].model, 'head')
            else:
                delattr(self.base_model[m], 'head')

    def _prepare_tbn(self):

        self._remove_classfication_layer()

    def _prepare_base_model(self, base_model):

        if base_model == 'ViT':
            self.base_model = OrderedDict()
            self.input_size = OrderedDict()
            self.input_mean = OrderedDict()
            self.input_std = OrderedDict()

            for m in self.modality:
                if m == "RGB":
                    self.base_model[m] = TimeSformer(img_size=224, num_classes=32, num_frames=self.num_segments, attention_type='divided_space_time',         
                                        pretrained_model='')
                else:
                    self.base_model[m] = create_model(num_classes=1000)
                    self.load_pretrain(m)

                self.input_size[m] = 224
                self.input_std[m] = [.229, .224, .225]

                if m == 'RGB':
                    self.input_mean[m] = [.485, .456, .406]

            self.feature_dim = 768
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))


    def forward(self, input):
        features = {}
        # Get the output for each modality
        for m in self.modality:
            if (m == 'RGB'):
                channel = 3
            elif (m == 'Accespec' or m == 'Gyrospec'):
                channel = 3

            base_model = getattr(self, m.lower())
            if m == 'RGB':
                 base_out = base_model(
                        input[m].view((-1, channel, self.num_segments) + input[m].size()[-2:]))
            else:
                 base_out = base_model(input[m])

            features[m] = base_out

        output = features

        return output

    def load_pretrain(self, modality):
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            model_dir='/data1/whx/PyCIL-ViT/pretrain',
            map_location="cpu", check_hash=True)
        state_dict = checkpoint["model"]
        self.base_model[modality].load_state_dict(state_dict)

    def freeze(self):
        for m in self.modality:
            logging.info('Freezing ' + m + ' stream\'s parameters')
            base_model = getattr(self, m.lower())
            for param in base_model.parameters():
                param.requires_grad_(False)

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        scale_size = {k: v * 256 // 224 for k, v in self.input_size.items()}
        return scale_size

    def get_augmentation(self):
        augmentation = {}
        if 'RGB' in self.modality:
            augmentation['RGB'] = torchvision.transforms.Compose(
                [GroupMultiScaleCrop(self.input_size['RGB'], [1, .875, .75, .66]),
                 GroupRandomHorizontalFlip(is_flow=False)])
        if 'Flow' in self.modality:
            augmentation['Flow'] = torchvision.transforms.Compose(
                [GroupMultiScaleCrop(self.input_size['Flow'], [1, .875, .75]),
                 GroupRandomHorizontalFlip(is_flow=True)])
        if 'RGBDiff' in self.modality:
            augmentation['RGBDiff'] = torchvision.transforms.Compose(
                [GroupMultiScaleCrop(self.input_size['RGBDiff'], [1, .875, .75]),
                 GroupRandomHorizontalFlip(is_flow=False)])

        return augmentation