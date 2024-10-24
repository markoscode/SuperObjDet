from collections import defaultdict
from .stiching_layers.stitching_layers import STITCH_LAYERS, SimpleStitchMoE
import torch
import torch.nn as nn
import numpy as np
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from .swin import SwinTransformer

def unpaired_stitching(front_depth=12, end_depth=24, idx_limit=None):
    num_stitches = front_depth

    block_ids = torch.tensor(list(range(front_depth)))
    block_ids = block_ids[None, None, :].float()
    end_mapping_ids = torch.nn.functional.interpolate(block_ids, end_depth)
    end_mapping_ids = end_mapping_ids.squeeze().long().tolist()
    front_mapping_ids = block_ids.squeeze().long().tolist()

    stitch_cfgs = []
    for idx in front_mapping_ids:
        for i, e_idx in enumerate(end_mapping_ids):
            if idx != e_idx or idx >= i:
                continue
            else:
                if idx_limit is not None and i >= idx_limit:
                    continue
                stitch_cfgs.append((idx, i))
    return stitch_cfgs, end_mapping_ids, num_stitches

# define a function to stitch by taking in the accuracies of probes
# TODO: DS

def new_unpaired_stitching(front_depth=12, end_depth=24):
    block_ids = torch.tensor(list(range(front_depth)))
    block_ids = block_ids[None, None, :].float()
    end_mapping_ids = torch.nn.functional.interpolate(block_ids, end_depth)
    print(f"End mapping ids: {end_mapping_ids}")
    end_mapping_ids = end_mapping_ids.squeeze().long().tolist()
    front_mapping_ids = block_ids.squeeze().long().tolist()

    stitch_cfgs = []
    for idx in front_mapping_ids:
        for i, e_idx in enumerate(end_mapping_ids):
            if idx != e_idx or idx >= i:
                continue
            else:
                if i == 0:
                    continue
                stitch_cfgs.append((idx, i))
    return stitch_cfgs, list(range(len(stitch_cfgs))), len(stitch_cfgs)

def paired_stitching(depth=12, kernel_size=2, stride=1):
    blk_id = list(range(depth))
    i = 0
    stitch_cfgs = []
    stitch_id = -1
    stitching_layers_mappings = []

    while i < depth:
        ids = blk_id[i:i + kernel_size]
        has_new_stitches = False
        for j in ids:
            for k in ids:
                if (j, k) not in stitch_cfgs:
                    if j >= k:
                        continue
                    has_new_stitches = True
                    stitch_cfgs.append((j, k))
                    stitching_layers_mappings.append(stitch_id + 1)

        if has_new_stitches:
            stitch_id += 1

        i += stride

    num_stitches = stitch_id + 1
    return stitch_cfgs, stitching_layers_mappings, num_stitches

def get_stitch_configs(depths, stage_id, new_stitch=False, comb_id=None):
    depths = sorted(depths)

    d = depths[0]
    total_configs = []
    total_stitches = []

    for i in range(1, len(depths)):
        next_d = depths[i]
        if next_d == d:
            stitch_cfgs, layers_mappings, num_stitches = paired_stitching(d)
        else:
            if new_stitch:
                stitch_cfgs, layers_mappings, num_stitches = new_unpaired_stitching(d, next_d)
            else:
                stitch_cfgs, layers_mappings, num_stitches = unpaired_stitching(d, next_d)
        if comb_id is None:
            comb = (i-1, i)
        else:
            comb = comb_id
        for cfg, layer_mapping_id in zip(stitch_cfgs, layers_mappings):
            total_configs.append({
                'comb_id': comb,
                'stage_id': stage_id,
                'stitch_cfgs': [cfg],
                'stitch_layers': [layer_mapping_id]
            })
        total_stitches.append((num_stitches, comb))
        d = next_d

    return total_configs, total_stitches

def rearrange_activations(activations):
    n_channels = activations.shape[-1]
    activations = activations.reshape(-1, n_channels)
    return activations

def ps_inv(x1, x2):
    '''Least-squares solver given feature maps from two anchors.
    
    Source: https://github.com/renyi-ai/drfrankenstein/blob/main/src/comparators/compare_functions/ps_inv.py
    '''
    x1 = rearrange_activations(x1)
    x2 = rearrange_activations(x2)

    if not x1.shape[0] == x2.shape[0]:
        raise ValueError('Spatial size of compared neurons must match when ' \
                         'calculating psuedo inverse matrix.')

    # Get transformation matrix shape
    shape = list(x1.shape)
    shape[-1] += 1

    # Calculate pseudo inverse
    x1_ones = torch.ones(shape)
    x1_ones[:, :-1] = x1
    A_ones = torch.matmul(torch.linalg.pinv(x1_ones), x2.to(x1_ones.device)).T

    # Get weights and bias
    w = A_ones[..., :-1]
    b = A_ones[..., -1]

    return w, b

@BACKBONE_REGISTRY.register()
class SNNet(Backbone):
    '''
    Stitchable Neural Networks
    '''

    def __init__(self, cfg, input_shape):
        in_chans = 3
        norm_layer = nn.LayerNorm
        anchor_names = ["TINY", "SMALL", "BASE"]
        anchors = []
        stiching_layer_type = cfg.SNNET.LAYER_TYPE if cfg.SNNET.LAYER_TYPE else "fc"
        new_stitch = None
        super(SNNet, self).__init__()
        for anc_name in anchor_names:
            pretrain_img_size = cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE
            patch_size = cfg.MODEL.SWIN.PATCH_SIZE
            embed_dim = cfg.MODEL.SWIN[anc_name].EMBED_DIM
            depths = cfg.MODEL.SWIN[anc_name].DEPTHS
            num_heads = cfg.MODEL.SWIN[anc_name].NUM_HEADS
            window_size = cfg.MODEL.SWIN[anc_name].WINDOW_SIZE
            mlp_ratio = cfg.MODEL.SWIN.MLP_RATIO
            qkv_bias = cfg.MODEL.SWIN.QKV_BIAS
            qk_scale = cfg.MODEL.SWIN.QK_SCALE
            drop_rate = cfg.MODEL.SWIN.DROP_RATE
            attn_drop_rate = cfg.MODEL.SWIN[anc_name].ATTN_DROP_RATE
            drop_path_rate = cfg.MODEL.SWIN.DROP_PATH_RATE
            ape = cfg.MODEL.SWIN[anc_name].APE
            patch_norm = cfg.MODEL.SWIN[anc_name].PATCH_NORM
            use_checkpoint = cfg.MODEL.SWIN.USE_CHECKPOINT
            anchors.append(SwinTransformer(
                pretrain_img_size,
                patch_size,
                in_chans,
                embed_dim,
                depths,
                num_heads,
                window_size,
                mlp_ratio,
                qkv_bias,
                qk_scale,
                drop_rate,
                attn_drop_rate,
                drop_path_rate,
                norm_layer,
                ape,
                patch_norm,
                use_checkpoint=use_checkpoint,
            ))
        #STICHNET SPECIFIC PARAMETERS
        #set to "fc" by defualt?
        
        

        
        self._out_features = cfg.MODEL.SWIN.OUT_FEATURES

        '''self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        # Need to check this out since we have different anchors now?
        self._out_feature_channels = {
            "res2": self.num_features[0],
            "res3": self.num_features[1],
            "res4": self.num_features[2],
            "res5": self.num_features[3],
        }'''

        self.anchors = nn.ModuleList(anchors) # list of anchors
        stage_depths = [anc.depths for anc in self.anchors]
        self.new_stitch = new_stitch

        total_configs = []
        self.num_stitches = []
        self.stitch_layers = nn.ModuleList()
        self.stitching_map_id = {}
        stitch_layer_cls = STITCH_LAYERS[stiching_layer_type]
        self.stitch_layer_type = stiching_layer_type
        self.stitch_configs = {}
        self.num_configs = 0
        
        for i in range(len(self.anchors)):
            total_configs.append({
                'comb_id': [i],
                'stitch_cfgs': [],
                'stitch_layers': []
            })
        '''for i in range(len(self.anchors)):
            if config is not None and (not config.TRAIN.FULL_TUNE) and (not config.EVAL_MODE):
                tune_layer = config.TRAIN.TUNE_LAYERS.split("#")
                if tune_layer[0] == "stitch":
                    continue
                elif tune_layer[0] == "anchor_stitch":
                    anchors_idx = [int(v) for v in tune_layer[1].split(",")]
                    if i in anchors_idx:
                        total_configs.append({
                            'comb_id': [i],
                            'stitch_cfgs': [],
                            'stitch_layers': []
                        })
                        print(f"--- Adding stitch config for anchor {i}")
                elif tune_layer[0] == "dynamic_anchor_stitch":
                    total_configs.append({
                        'comb_id': [i],
                        'stitch_cfgs': [],
                        'stitch_layers': []
                    })
            else:
                total_configs.append({
                    'comb_id': [i],
                    'stitch_cfgs': [],
                    'stitch_layers': []
                })'''

        # iterate through all stages
        for i in range(4):
            
            # skip the last stage
            if i == 3:
                continue
                
            cur_depths = [stage_depths[anc_id][i] for anc_id in range(len(self.anchors))] # depths of ti, s & b for stage i, so [6,18,18] for i=2
            stage_configs, stage_stitches = get_stitch_configs(cur_depths, i, new_stitch=self.new_stitch)
            self.num_stitches.append(stage_stitches)
            total_configs += stage_configs
            stage_stitching_layers = nn.ModuleList()

            for j, (num_s, comb) in enumerate(stage_stitches):
                front, end = comb
                stage_stitching_layers.append(nn.ModuleList(
                    [stitch_layer_cls(self.anchors[front].stage_dims[i], self.anchors[end].stage_dims[i]) for _ in range(num_s)]))
                self.stitching_map_id[f'{i}-{front}-{end}'] = j
            
            self.stitch_layers.append(stage_stitching_layers)

        self.stitch_configs = {i: cfg for i, cfg in enumerate(total_configs)}
        # self.num_configs = len(total_configs)
        # self.stitch_config_id = 0
    
        # hardcoding for probenets
        # obtained the following indices from probe_visualizations.ipynb
        new_stitches = [(0, 2), (0, 3), (1, 4), (1, 5), (1, 6), (2, 7), (2, 8), (2, 9), (4, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15), (3, 16)]
        new_total_configs = []
        j = 0
        for key in self.stitch_configs:
            if 'stage_id' in self.stitch_configs[key]:
                if self.stitch_configs[key]['stage_id'] == 2 and self.stitch_configs[key]['comb_id'] == (0,1):
                    if j>=len(new_stitches):
                        continue
                    entry = self.stitch_configs[key]
                    entry["stitch_cfgs"] = [new_stitches[j]]
                    new_total_configs.append(entry)
                    j += 1
                else:
                    new_total_configs.append(self.stitch_configs[key])
            else:
                new_total_configs.append(self.stitch_configs[key])
        
        self.stitch_configs = {i: cfg for i, cfg in enumerate(new_total_configs)}
        self.num_configs = len(new_total_configs)
        self.stitch_config_id = 0
        self._size_devisibility = 32

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def sample_random_stitch_config(self):
        self.stitch_cfg_id = np.random.randint(0, self.num_configs)
        return self.stitch_cfg_id

    def set_stitch_id(self, stitch_config_id):
        self.stitch_config_id = stitch_config_id

    def initialize_stitching_weights(self, x, layer_id=None):
        anchor_features = []
        with torch.no_grad():
            for anc in self.anchors:
                anchor_features.append(anc.extract_block_features(x))

        for stage_id in range(4):
            if stage_id == 3:
                break
            stage_stitches = self.num_stitches[stage_id]

            for j, (num_s, comb) in enumerate(stage_stitches):
                front, end = comb
                stitching_dicts = defaultdict(set)
                for id, config in self.stitch_configs.items():
                    if config['comb_id'] == comb and stage_id == config['stage_id']:
                        stitching_dicts[config['stitch_layers'][0]].add(config['stitch_cfgs'][0])

                for stitch_layer_id, stitch_positions in stitching_dicts.items():
                    weight_candidates = []
                    bias_candidates = []
                    for front_id, end_id in stitch_positions:
                        front_blk_feat = anchor_features[front][stage_id][front_id]
                        end_blk_feat = anchor_features[end][stage_id][end_id - 1]
                        w, b = ps_inv(front_blk_feat, end_blk_feat)
                        weight_candidates.append(w)
                        bias_candidates.append(b)
                    weights = torch.stack(weight_candidates).mean(dim=0)
                    bias = torch.stack(bias_candidates).mean(dim=0)
                    stitch_layer =  self.stitch_layers[stage_id][j][stitch_layer_id]
                    if isinstance(stitch_layer, SimpleStitchMoE) and layer_id is not None:
                          stitch_layer.init_stitch_weights_bias(weights, bias, layer_id=layer_id)
                    else:
                        stitch_layer.init_stitch_weights_bias(weights, bias)
                    print(f'Initialized Stitching Model {front} to Model {end}, Stage {stage_id}, Layer {stitch_layer_id}')


    def get_model_size(self, stitch_cfg_id):
        comb_id = self.stitch_configs[stitch_cfg_id]['comb_id']
        if len(comb_id) == 1:
            return sum(p.numel() for p in self.anchors[comb_id[0]].parameters())

        stitch_cfgs = self.stitch_configs[stitch_cfg_id]['stitch_cfgs']
        stitch_stage_id = self.stitch_configs[stitch_cfg_id]['stage_id']
        stitch_layer_ids = self.stitch_configs[stitch_cfg_id]['stitch_layers']

        cfg = stitch_cfgs[0]
        total_params = 0
        total_params += self.anchors[comb_id[0]].get_model_size_util(stage_id=stitch_stage_id, blk_id = cfg[0])

        sl_id = stitch_layer_ids[0]
        key = f'{stitch_stage_id}-{comb_id[0]}-{comb_id[1]}'
        stitch_projection_id = self.stitching_map_id[key]
        total_params += sum(p.numel() for p in self.stitch_layers[stitch_stage_id][stitch_projection_id][sl_id].parameters())


        total_params += self.anchors[comb_id[1]].get_model_size_from(stage_id=stitch_stage_id, blk_id=cfg[1])

        return total_params
    
    def forward(self, x):
        '''if self.training:
            stitch_cfg_id = np.random.randint(0, self.num_configs)
        else:
            stitch_cfg_id = self.stitch_config_id'''
        stitch_cfg_id = self.stitch_config_id

        #if return_activation:
        #    input_tensor = x

        comb_id = self.stitch_configs[stitch_cfg_id]['comb_id']
        # print(f"Len combid : {len(comb_id)}")
        if len(comb_id) == 1:
            out = self.anchors[comb_id[0]](x)
            #if return_activation:
            #    return out, None, None
            return out

        stitch_cfgs = self.stitch_configs[stitch_cfg_id]['stitch_cfgs']
        stitch_stage_id = self.stitch_configs[stitch_cfg_id]['stage_id']
        stitch_layer_ids = self.stitch_configs[stitch_cfg_id]['stitch_layers']

        cfg = stitch_cfgs[0]

        x, outs = self.anchors[comb_id[0]].forward_until(x, stage_id=stitch_stage_id, blk_id=cfg[0])
        #if return_activation:
        #    next_anchor_activation = self.anchors[comb_id[1]].forward_until(input_tensor, stage_id=stitch_stage_id, blk_id=(cfg[1]-1))

        sl_id = stitch_layer_ids[0]
        key = f'{stitch_stage_id}-{comb_id[0]}-{comb_id[1]}'
        stitch_projection_id = self.stitching_map_id[key]
        # if return_activation:
        #     prj_input = x.detach()
        #     projected_activation = self.stitch_layers[stitch_stage_id][stitch_projection_id][sl_id](prj_input)
            
        x = self.stitch_layers[stitch_stage_id][stitch_projection_id][sl_id](x)
        #if return_activation:
        #    projected_activation = x

        x = self.anchors[comb_id[1]].forward_from(x, stage_id=stitch_stage_id, blk_id=cfg[1], outs=outs)
        #if return_activation:
            # print(f"returning x:{x.shape} prj_act:{projected_activation.shape} next_anchor_active: {next_anchor_activation.shape} ")
        #    return x, projected_activation, next_anchor_activation
        #else:
        return x
