import torch
from torch import nn
import torch.nn.functional as F
from model.Transformer import Transformer
import model.resnet as models
import model.vgg as vgg_models
from model.PSPNet import OneModel as PSPNet
from einops import rearrange
import os

from torchvision.transforms import Resize
from .clip import *
clip_path = 'clip'
semantic_backbone = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']

'''
Version: [+SSA] Add Salient Semantic Alignment module 
         [+SMF] Add Saliency Map Filter
'''


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def get_similarity(q, s, mask):
    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)
    # mask = F.interpolate((mask == 1).float(), q.shape[-2:])
    mask = F.interpolate((mask >= 0.0).float(), q.shape[-2:])

    # mask = F.interpolate((mask >= 0.5).float(), q.shape[-2:])

    cosine_eps = 1e-7
    s = s * mask
    bsize, ch_sz, sp_sz, _ = q.size()[:]
    tmp_query = q
    tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
    tmp_query_norm = torch.norm(tmp_query, 2, 1, True)
    tmp_supp = s
    tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1).contiguous()
    tmp_supp = tmp_supp.contiguous().permute(0, 2, 1).contiguous()
    tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)
    similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
    similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
    similarity = similarity.view(bsize, 1, sp_sz, sp_sz)
    return similarity


def get_gram_matrix(fea):
    b, c, h, w = fea.shape
    fea = fea.reshape(b, c, h * w)  # C*N
    fea_T = fea.permute(0, 2, 1)  # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T) / (torch.bmm(fea_norm, fea_T_norm) + 1e-7)  # C*C
    return gram


def get_vgg16_layer(model):
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4


import torch
import torch.nn.functional as F


def saliency_local_peaks(saliency_map: torch.Tensor, block_num=4):
    """
    saliency map ---> block_num x block_num
    :param saliency_map: [B, 1, H, W]
    :param block_num: How many pieces per side
    :return: [B, block_num*block_num, 2] tensor, (y, x)
    """
    B, _, H, W = saliency_map.shape
    device = saliency_map.device

    #
    new_H = (H // block_num) * block_num
    new_W = (W // block_num) * block_num
    saliency_map = F.interpolate(saliency_map, size=(new_H, new_W), mode='bilinear', align_corners=False)

    # reshape成 [B, 1, block_num, block_h, block_num, block_w]
    saliency_map = saliency_map.view(B, 1, block_num, new_H // block_num, block_num, new_W // block_num)

    saliency_map = saliency_map.permute(0, 2, 4, 1, 3, 5).contiguous()
    # [B, block_num, block_num, 1, block_h, block_w]

    saliency_map = saliency_map.view(B, block_num * block_num, -1)  # [B, n_blocks, block_size]

    # in block: max and min
    top_val, top_idx = saliency_map.max(dim=2)  # top_val: [B, n_blocks]

    sal_center_vals = top_val
    return sal_center_vals


class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()

        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.dataset = args.data_set
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60
        self.low_fea_id = args.low_fea[-1]

        assert args.layers in [50, 101, 152]
        from torch.nn import BatchNorm2d as BatchNorm

        # self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.criterion = nn.BCELoss()

        self.shot = args.shot
        self.vgg = args.vgg
        models.BatchNorm = BatchNorm

        PSPNet_ = PSPNet(args)

        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        weight_path = os.path.join(BASE_DIR, args.pre_weight)

        new_param = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']

        try:
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4
        self.ppm = PSPNet_.ppm
        self.cls = nn.Sequential(PSPNet_.cls[0], PSPNet_.cls[1])
        self.base_learnear = nn.Sequential(PSPNet_.cls[2], PSPNet_.cls[3], PSPNet_.cls[4])

        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512

        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.query_merge = nn.Sequential(
            nn.Conv2d(512 + 2, 64, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        self.supp_merge = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        self.transformer = Transformer(shot=self.shot)

        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.gram_merge.weight))
        # Learner Ensemble
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.cls_merge.weight))

        self.cls_merge2 = nn.Conv2d(2, 1, kernel_size=1, bias=False) # new add

        self.cls_merge3 = nn.Conv2d(2, 1, kernel_size=1, bias=False) # new add

        # semantic
        self.clip_model, _ = clip.load(semantic_backbone[4])
        self.clip_model = self.clip_model.eval()

        # K-Shot Reweighting
        if args.shot > 1:
            self.kshot_trans_dim = args.kshot_trans_dim
            if self.kshot_trans_dim == 0:
                self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)
                self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
            else:
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y_m=None, y_b=None, s_x=None, s_y=None, cat_idx=None):
        h, w = x.shape[-2:]
        _, _, query_feat_2, query_feat_3, query_feat_4, query_feat_5 = self.extract_feats(x)

        if self.vgg:
            query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2), query_feat_3.size(3)),
                                         mode='bilinear', align_corners=True)
        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        mask = rearrange(s_y, "b n h w -> (b n) 1 h w")
        # mask = (mask == 1).float()
        # mask = (mask >= 0.0).float()
        # mask = (mask >= 0.5).float()
        # ========================================
        # [SMF] Saliency Map Filter
        # [s]=====================================
        b, c, h, w = s_y.shape
        sal_center_vals = saliency_local_peaks(s_y)
        threshold = 0.1
        real_sal_center = (sal_center_vals > threshold).float()
        selected_vals = sal_center_vals * real_sal_center
        total_sum = selected_vals.sum(dim=1)
        total_count = real_sal_center.sum(dim=1)

        # filter_threshold = total_sum / (total_count + 1e-6)
        filter_threshold = torch.norm(selected_vals.view(selected_vals.size(0), -1), p=2, dim=1) / total_count

        filter_threshold = filter_threshold.view(b, 1, 1, 1).expand(b, -1, h, w)
        mask = torch.where(s_y > filter_threshold, s_y, torch.tensor(0.0))
        # [e]=====================================


        s_x = rearrange(s_x, "b n c h w -> (b n) c h w")
        supp_feat_0, supp_feat_1, supp_feat_2, supp_feat_3, supp_feat_4, supp_feat_5 = self.extract_feats(s_x, mask)
        if self.vgg:
            supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                        align_corners=True)
        supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
        supp_feat = self.down_supp(supp_feat)
        supp_feat_bin = Weighted_GAP(supp_feat, \
                                     F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                                   mode='bilinear', align_corners=True))
        supp_feat_bin = supp_feat_bin.repeat(1, 1, supp_feat.shape[-2], supp_feat.shape[-1])
        supp_feat_item = eval('supp_feat_' + self.low_fea_id)
        supp_feat_item = rearrange(supp_feat_item, "(b n) c h w -> b n c h w", n=self.shot)
        supp_feat_list = [supp_feat_item[:, i, ...] for i in range(self.shot)]

        if self.shot == 1:
            # similarity2 = get_similarity(query_feat_4, supp_feat_4, s_y)
            # similarity1 = get_similarity(query_feat_5, supp_feat_5, s_y)
            similarity2 = get_similarity(query_feat_4, supp_feat_4, mask)
            similarity1 = get_similarity(query_feat_5, supp_feat_5, mask)
        else:
            mask = rearrange(mask, "(b n) c h w -> b n c h w", n=self.shot)
            supp_feat_4 = rearrange(supp_feat_4, "(b n) c h w -> b n c h w", n=self.shot)
            supp_feat_5 = rearrange(supp_feat_5, "(b n) c h w -> b n c h w", n=self.shot)
            similarity1 = [get_similarity(query_feat_5, supp_feat_5[:, i, ...], mask=mask[:, i, ...]) for i in
                           range(self.shot)]
            similarity2 = [get_similarity(query_feat_4, supp_feat_4[:, i, ...], mask=mask[:, i, ...]) for i in
                           range(self.shot)]
            mask = rearrange(mask, "b n c h w -> (b n) c h w")
            supp_feat_4 = rearrange(supp_feat_4, "b n c h w -> (b n) c h w")
            supp_feat_5 = rearrange(supp_feat_5, "b n c h w -> (b n) c h w")
            similarity2 = torch.stack(similarity2, dim=1).mean(1)
            similarity1 = torch.stack(similarity1, dim=1).mean(1)
        similarity = torch.cat([similarity1, similarity2], dim=1)

        supp_feat = self.supp_merge(torch.cat([supp_feat, supp_feat_bin], dim=1))
        supp_feat_bin = rearrange(supp_feat_bin, "(b n) c h w -> b n c h w", n=self.shot)
        supp_feat_bin = torch.mean(supp_feat_bin, dim=1)
        query_feat = self.query_merge(torch.cat([query_feat, supp_feat_bin, similarity * 10], dim=1))

        meta_out, weights = self.transformer(query_feat, supp_feat, mask, similarity)
        base_out = self.base_learnear(query_feat_5)

        meta_out_soft = meta_out.softmax(1)
        base_out_soft = base_out.softmax(1)

        # K-Shot Reweighting
        bs = x.shape[0]
        que_gram = get_gram_matrix(eval('query_feat_' + self.low_fea_id))  # [bs, C, C] in (0,1)
        norm_max = torch.ones_like(que_gram).norm(dim=(1, 2))
        est_val_list = []
        for supp_item in supp_feat_list:
            supp_gram = get_gram_matrix(supp_item)
            gram_diff = que_gram - supp_gram
            est_val_list.append((gram_diff.norm(dim=(1, 2)) / norm_max).reshape(bs, 1, 1, 1))  # norm2
        est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
        if self.shot > 1:
            val1, idx1 = est_val_total.sort(1)
            val2, idx2 = idx1.sort(1)
            weight = self.kshot_rw(val1)
            idx3 = idx1.gather(1, idx2)
            weight = weight.gather(1, idx3)
            weight_soft = torch.softmax(weight, 1)
        else:
            weight_soft = torch.ones_like(est_val_total)
        est_val = (weight_soft * est_val_total).sum(1, True)  # [bs, 1, 1, 1]

        # Following the implementation of BAM ( https://github.com/chunbolang/BAM )
        meta_map_bg = meta_out_soft[:, 0:1, :, :]
        meta_map_fg = meta_out_soft[:, 1:, :, :]

        # if self.training and self.cls_type == 'Base':
        #     c_id_array = torch.arange(self.base_classes + 1, device='cuda')
        #     base_map_list = []
        #     for b_id in range(bs):
        #         c_id = cat_idx[0][b_id] + 1
        #         c_mask = (c_id_array != 0) & (c_id_array != c_id)
        #         base_map_list.append(base_out_soft[b_id, c_mask, :, :].unsqueeze(0).sum(1, True))
        #     base_map = torch.cat(base_map_list, 0)
        # else:
        #     base_map = base_out_soft[:, 1:, :, :].sum(1, True)

        base_map = base_out_soft[:, 1:, :, :].sum(1, True)

        meta_out = self.cls_merge3(meta_out)  # new add
        # meta_out = meta_map_fg.sum(1, True)

        est_map = est_val.expand_as(meta_map_fg)

        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))

        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)  # [bs, 1, 60, 60]

        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)
        final_out = self.cls_merge2(final_out)

        meta_out = self.sigmoid(meta_out)
        base_map = self.sigmoid(base_map)
        final_out = self.sigmoid(final_out)

        # Output Part
        meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
        base_map = F.interpolate(base_map, size=(h, w), mode='bilinear', align_corners=True)
        final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)

        # ========================================
        # [SSA] Salient Semantic Alignment module
        # [s]=====================================

        sal_sup = s_y * (s_x.squeeze(1))
        sal_que = final_out * x

        # alpha = 0.5
        # sal_sup = alpha * (s_y * (s_x.squeeze(1))) + (1 - alpha) * s_x
        # sal_que = alpha * final_out * x + (1 - alpha) * x

        torch_resize = Resize([224, 224])
        sal_sup = torch_resize(sal_sup)
        sal_que = torch_resize(sal_que)
        with torch.no_grad():
            sem_sal_sup = self.clip_model.encode_image(sal_sup)
            sem_sal_que = self.clip_model.encode_image(sal_que)
        # [e]=====================================

        y_m = y_m.unsqueeze(1)
        y_b = y_b.unsqueeze(1)

        if self.training:
            assert torch.all((y_m >= 0) & (y_m <= 1)), f"y_m --> over range: min={y_m.min()}, max={y_m.max()}"
            assert torch.all((y_b >= 0) & (y_b <= 1)), f"y_b --> over range: min={y_b.min()}, max={y_b.max()}"

            main_loss = self.criterion(final_out, y_m)
            aux_loss1 = self.criterion(meta_out, y_m)
            aux_loss2 = self.criterion(base_map, y_b)
            aux_loss3 = self.cosine_loss(sem_sal_sup, sem_sal_que)

            y_m = y_m.squeeze(1)

            weight_t = y_m

            for i, weight in enumerate(weights):
                assert torch.all((weight >= 0) & (weight <= 1)), f"weight --> over range: min={weight.min()}, max={weight.max()}"
                assert torch.all((weight_t >= 0) & (weight_t <= 1)), f"weight_t --> over range: min={weight_t.min()}, max={weight_t.max()}"
                if i == 0:
                    distil_loss = self.disstil_loss(weight_t, weight)
                else:
                    distil_loss += self.disstil_loss(weight_t, weight)
                weight_t = weight.detach()

            return final_out, main_loss + aux_loss1, distil_loss / 3, aux_loss2 + aux_loss3
        else:
            return final_out, meta_out, base_map

    def cosine_loss(self, x1, x2):
        return 1 - F.cosine_similarity(x1, x2, dim=-1).mean()

    def disstil_loss(self, t, s):
        if t.shape[-2:] != s.shape[-2:]:
            t = F.interpolate(t.unsqueeze(1), size=s.shape[-2:], mode='bilinear').squeeze(1)

        loss = self.criterion(s, t)
        return loss

    def get_optim(self, model, args, LR):
        optimizer = torch.optim.AdamW(
            [
                {'params': model.transformer.mix_transformer.parameters()},
                {'params': model.down_supp.parameters(), "lr": LR * 10},
                {'params': model.down_query.parameters(), "lr": LR * 10},
                {'params': model.supp_merge.parameters(), "lr": LR * 10},
                {'params': model.query_merge.parameters(), "lr": LR * 10},
                {'params': model.gram_merge.parameters(), "lr": LR * 10},
                {'params': model.cls_merge.parameters(), "lr": LR * 10},
                {'params': model.cls_merge2.parameters(), "lr": LR * 10},
                {'params': model.cls_merge3.parameters(), "lr": LR * 10}
            ], lr=LR, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        return optimizer

    def freeze_modules(self, model):
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.ppm.parameters():
            param.requires_grad = False
        for param in model.cls.parameters():
            param.requires_grad = False
        for param in model.base_learnear.parameters():
            param.requires_grad = False
        for param in model.clip_model.parameters():
            param.requires_grad = False

    def extract_feats(self, x, mask=None):
        results = []
        with torch.no_grad():
            if mask is not None:
                tmp_mask = F.interpolate(mask, size=x.shape[-2], mode='nearest')
                x = x * tmp_mask
            feat = self.layer0(x)
            results.append(feat)
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for _, layer in enumerate(layers):
                feat = layer(feat)
                results.append(feat.clone())
            feat = self.ppm(feat)
            feat = self.cls(feat)
            results.append(feat)
        return results
