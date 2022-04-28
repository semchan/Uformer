import logging

import torch

from anchor_free import anchor_free_helper
from anchor_free.Uformer import UTransformer
from anchor_free.losses import calc_ctr_loss, calc_cls_loss, calc_loc_loss,calc_rec_loss
from evaluate import evaluate
from helpers import data_helper, vsumm_helper

logger = logging.getLogger()


def train(args, split, save_path):
    model = UTransformer(dim_in=args.num_feature,
                         dim_out=args.num_feature,
                         heads=8,
                         mlp_dim=16,
                         dropout_rate=0.3,
                         attn_dropout_rate=0.3)

    print('anchor_free DSNet mode total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    model = model.to(args.device)

    model.train()

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
        stats = data_helper.AverageMeter('loss', 'cls_loss', 'loc_loss','ctr_loss','rec_loss')

        for _, seq, seqdiff,gtscore, change_points, n_frames, nfps, picks, _ in train_loader:
            keyshot_summ = vsumm_helper.get_keyshot_summ(
                gtscore, change_points, n_frames, nfps, picks)
            target = vsumm_helper.downsample_summ(keyshot_summ)

            if not target.any():
                continue

            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(args.device)

            cls_label = target#True, False
            loc_label = anchor_free_helper.get_loc_label(target)
            ctr_label = anchor_free_helper.get_ctr_label(target, loc_label)# score

            cls_label = torch.tensor(cls_label, dtype=torch.float32).to(args.device)
            loc_label = torch.tensor(loc_label, dtype=torch.float32).to(args.device)
            ctr_label = torch.tensor(ctr_label, dtype=torch.float32).to(args.device)
            seqdiff= torch.tensor(seqdiff, dtype=torch.float32).unsqueeze(0).to(args.device)
                        # pred_cls, pred_loc, pred_ctr = model(seq)
            # out = model(seq)
            out = model(seq,seqdiff)





            output_dict = out['d0']
            pred_cls = output_dict['pred_cls']
            pred_loc = output_dict['pred_loc']
            pred_ctr = output_dict['pred_ctr']
            #reconstructions = output_dict['reconstructions']

            cls_loss = calc_cls_loss(pred_cls, cls_label, args.cls_loss)
            loc_loss = calc_loc_loss(pred_loc, loc_label, cls_label,
                                     args.reg_loss)
            ctr_loss = calc_ctr_loss(pred_ctr, ctr_label, cls_label)



            #rec_loss = calc_rec_loss(reconstructions, seq)
            rec_loss=ctr_loss
            loss = cls_loss + args.lambda_reg * loc_loss + args.lambda_ctr * ctr_loss#+rec_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats.update(loss=loss.item(), cls_loss=cls_loss.item(),
                         loc_loss=loc_loss.item(), rec_loss=rec_loss.item(), ctr_loss=ctr_loss.item())

        val_fscore, _ = evaluate(model, val_loader, args.nms_thresh, args.device)

        if max_val_fscore < val_fscore:
            max_val_fscore = val_fscore
            torch.save(model.state_dict(), str(save_path))

        logger.info(f'Epoch: {epoch}/{args.max_epoch} '
                    f'Loss: {stats.cls_loss:.4f}/{stats.loc_loss:.4f}/{stats.ctr_loss:.4f}/{stats.rec_loss:.4f}/{stats.loss:.4f} '
                    f'F-score cur/max: {val_fscore:.4f}/{max_val_fscore:.4f}')

    return max_val_fscore#
