import math
from collections import OrderedDict
from torch import nn

from ..config import Cfg
from ..misc.clip_pad import cat_crop
from .activation import activation_attr
from .common import NormCfg, UpSamplingCfg, DownSamplingCfg


@Cfg.register_obj('model', type='simple-unet')
class SimpleUnetCfg(Cfg.Obj):
    activation = activation_attr(default='relu')
    norm = Cfg.obj(None, obj_types=NormCfg, shortcut='norm')
    upsampling: UpSamplingCfg = Cfg.obj('bilinear', shortcut='type')
    downsampling: DownSamplingCfg = Cfg.obj('max', shortcut='type')
    dropout = Cfg.float(0, min=0, max=1)

    n_scale = Cfg.int(4, min=2)
    depth = Cfg.list(Cfg.int(min=1), default=2)
    n_features = Cfg.oneOf(Cfg.int(min=1), Cfg.list(Cfg.int(min=1)), default=16)

    kernel = Cfg.shape(3, dim=2)
    dilation = Cfg.int(1, min=1)
    padding = Cfg.oneOf('same', Cfg.int(min=0), default='same')
    padding_mode = Cfg.oneOf('zeros', 'reflect', 'replicate', 'circular', default='zeros')
    bias = Cfg.bool(None)

    @depth.post_checker
    @n_features.post_checker
    def check_specs(self, v):
        if isinstance(v, int):
            # n_features only
            return [v * 2 ** (s + 1) for s in range(self.n_scale)]
        if len(v) <= self.n_scale:
            return v + [v[-1]] * (self.n_scale - len(v))
        if len(v) <= self.n_scale * 2 - 1:
            return v + [v[-1]] * (self.n_scale * 2 - 1 - len(v))
        raise Cfg.InvalidAttr('Too many element in list',
                              f'Length of n-features and depth should not exceed {self.n_scale * 2 - 1} '
                              f'when n-scale is {self.n_scale}')

    @property
    def encoder_out_features_depth(self):
        return self.n_features[:self.n_scale], self.depth[:self.n_scale]

    @property
    def decoder_out_features_depth(self):
        def extract_decoder(spec):
            if len(spec) > self.n_scale:
                return spec[self.n_scale:]
            else:
                return list(reversed(spec[:-1]))

        return extract_decoder(self.n_features), extract_decoder(self.depth)

    @padding.post_checker
    def check_padding(self, p):
        if p == 'same':
            return math.ceil(self.kernel / 2)
        return p

    def create(self, in_channels: int):
        return SimpleUnet(self, in_channels)


class SimpleUnet(nn.Module):
    def __init__(self, cfg: SimpleUnetCfg, in_channels: int):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels

        n_in = in_channels

        encoder = OrderedDict()
        for n, (n_out, depth) in enumerate(zip(*self.cfg.encoder_out_features_depth)):
            encoder[f'Encoder-{n}'] = self.create_stage(n_in, n_out, depth, prepend='downsampling' if n > 0 else None)
            n_in = n_out
        self.encoder = nn.Sequential(encoder)

        decoder = OrderedDict()
        self.upsamplings = []
        for n, (n_out, depth) in enumerate(zip(*self.cfg.decoder_out_features_depth)):
            self.upsamplings += [self.cfg.upsampling.create(n_in)]
            self.add_module(f'Decoder-{n}-upsample', self.upsamplings[-1])

            decoder[f'Decoder-{n}'] = self.create_stage(n_in+n_out, n_out, depth)
            n_in = n_out
        self.decoder = nn.Sequential(decoder)

        final = OrderedDict()
        if self.cfg.norm:
            final['norm'] = self.cfg.norm.create(n_in)
        n_classes = self.cfg.root()['task'].n_classes
        if n_classes == 'binary':
            n_classes = 2
        final['conv'] = nn.Conv2d(n_in, n_classes, kernel_size=1)
        self.final = nn.Sequential(final)
        self.n_classes = n_classes

        self.dropout = nn.Dropout2d(self.cfg.dropout)

    def forward(self, x):
        x_stages = []
        for stage in self.encoder:
            x = stage(x)
            x_stages.append(self.dropout(x))
        x_stages.pop()

        x = self.dropout(x)

        for stage, upsampling in zip(self.decoder, self.upsamplings):
            x = upsampling(x)
            x_stage = x_stages.pop()
            x = cat_crop(x_stage, x)
            x = stage(x)
        y = self.final(x)
        return y

    def create_stage(self, n_in, n_out, depth, activation=True, prepend=None):
        cfg = self.cfg
        norm = cfg.norm
        bias = cfg.bias if cfg.bias is not None else norm is None

        stage = OrderedDict()
        if prepend == 'downsampling':
            stage['downsampling'] = self.cfg.downsampling.create(n_in, k=2)

        for d in range(depth):
            d = str(d).zfill(math.ceil(math.log10(depth)))
            if norm:
                stage['norm' + d] = norm.create(n_in)
            stage['conv' + d] = nn.Conv2d(n_in, n_out,
                                          kernel_size=cfg.kernel,
                                          dilation=cfg.dilation,
                                          bias=bias,
                                          padding=cfg.padding, padding_mode=cfg.padding_mode,
                                          )
            n_in = n_out
            if activation:
                stage['activation' + d] = cfg.activation.create()
        return nn.Sequential(stage)
