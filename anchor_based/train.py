import logging

import numpy as np
import torch
from torch import nn

from anchor_based import anchor_helper
from anchor_free.Uformer import UTransformerAB
from anchor_based.losses import calc_cls_loss, calc_loc_loss
from evaluate import evaluate
from helpers import data_helper, vsumm_helper, bbox_helper

logger = logging.getLogger()


def xavier_init(module):
    cls_name = module.__class__.__name__
    if 'Linear' in cls_name or 'Conv' in cls_name:
        nn.init.xavier_uniform_(module.weight, gain=np.sqrt(2.0))
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.1)


def train(args, split, save_path):
    model = UTransformerAB(anchor_scales=args.anchor_scales,
                         dim_in=args.num_feature,
                         dim_out=args.num_feature,
                         heads=8,
                         mlp_dim=16,
                         dropout_rate=0.3,
                         attn_dropout_rate=0.3)
    model = model.to(args.device)
    import time
    dummy_input = torch.randn((1, 100, 1024)).cuda()
    print('model　total: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.eval()
    t1 = time.time()
    for i in range(10000):
        torch.cuda.empty_cache()
        out=model(dummy_input,dummy_input)#[2, 100, 8,1];[2,400,100];[2,100]
        # out=model(dummy_input)#[2, 100, 8,1];[2,400,100];[2,100]
        # print(out)
    t2 = time.time()
    t = (t2-t1)/10000
    print('average time:',t)


    model.apply(xavier_init)
    print('args.weight_decay:',args.weight_decay)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr,
                                 weight_decay=args.weight_decay)

    max_val_fscore = -1

    train_set = data_helper.VideoDataset(split['train_keys'])
    train_loader = data_helper.DataLoader(train_set, shuffle=True)

    val_set = data_helper.VideoDataset(split['test_keys'])
    val_loader = data_helper.DataLoader(val_set, shuffle=False)

    for epoch in range(args.max_epoch):
        model.train()
        stats = data_helper.AverageMeter('loss', 'cls_loss', 'loc_loss')

        for _, seq, seqdiff,gtscore, cps, n_frames, nfps, picks, _ in train_loader:
            keyshot_summ = vsumm_helper.get_keyshot_summ(
                gtscore, cps, n_frames, nfps, picks)
            target = vsumm_helper.downsample_summ(keyshot_summ)

            if not target.any():
                continue

            # if target_size<target.size:
            #     target_size = target.size
            # print('target.size:',target_size)
            target_bboxes = bbox_helper.seq2bbox(target)
            target_bboxes = bbox_helper.lr2cw(target_bboxes)
            anchors = anchor_helper.get_anchors(target.size, args.anchor_scales)
            # Get class and location label for positive samples
            cls_label, loc_label = anchor_helper.get_pos_label(
                anchors, target_bboxes, args.pos_iou_thresh)

            # Get negative samples
            num_pos = cls_label.sum()
            cls_label_neg, _ = anchor_helper.get_pos_label(
                anchors, target_bboxes, args.neg_iou_thresh)
            cls_label_neg = anchor_helper.get_neg_label(
                cls_label_neg, int(args.neg_sample_ratio * num_pos))

            # Get incomplete samples
            cls_label_incomplete, _ = anchor_helper.get_pos_label(
                anchors, target_bboxes, args.incomplete_iou_thresh)
            cls_label_incomplete[cls_label_neg != 1] = 1
            cls_label_incomplete = anchor_helper.get_neg_label(
                cls_label_incomplete,
                int(args.incomplete_sample_ratio * num_pos))

            cls_label[cls_label_neg == -1] = -1
            cls_label[cls_label_incomplete == -1] = -1

            cls_label = torch.tensor(cls_label, dtype=torch.float32).to(args.device)
            loc_label = torch.tensor(loc_label, dtype=torch.float32).to(args.device)

            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(args.device)
            seqdiff= torch.tensor(seqdiff, dtype=torch.float32).unsqueeze(0).to(args.device)
            # pred_cls, pred_loc = model(seq)


            out = model(seq,seqdiff)

            output_dict = out['d0']
            pred_cls = output_dict['pred_cls']
            pred_loc = output_dict['pred_loc']

            loc_loss = calc_loc_loss(pred_loc, loc_label, cls_label)
            cls_loss = calc_cls_loss(pred_cls, cls_label)

            loss = cls_loss + args.lambda_reg * loc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats.update(loss=loss.item(), cls_loss=cls_loss.item(),
                         loc_loss=loc_loss.item())

        val_fscore, _ = evaluate(model, val_loader, args.nms_thresh, args.device)

        if max_val_fscore < val_fscore:
            max_val_fscore = val_fscore
            torch.save(model.state_dict(), str(save_path))
        lrate=optimizer.state_dict()['param_groups'][0]['lr']

        logger.info(f'Epoch: {epoch}/{args.max_epoch} '
                    f'Loss: {stats.cls_loss:.4f}/{stats.loc_loss:.4f}/{stats.loss:.4f} '
                    f'F-score cur/max: {val_fscore:.4f}/{max_val_fscore:.4f} '
                    f'lr: {lrate:.6f} ')

        # logger.info('Epoch: {epoch}/{args.max_epoch} '
        #             'Loss: {stats.cls_loss:.4f}/{stats.loc_loss:.4f}/{stats.loss:.4f} '
        #             'F-score cur/max: {val_fscore:.4f}/{max_val_fscore:.4f} '
        #             'lr: {lrate:.6f} ')

    return max_val_fscore
