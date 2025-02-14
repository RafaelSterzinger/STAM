r""" Visual Prompt Encoder training (validation) code """
import os
import argparse

import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributed as dist

from model.VRP_encoder import VRP_encoder
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from SAM2pred import SAM_pred


def train(args, epoch, model, sam_model, dataloader, optimizer, scheduler, training):
    r""" Train VRP_encoder model """

    utils.fix_randseed(args.seed + epoch) if training else utils.fix_randseed(args.seed)
    model.module.train_mode() if training else model.module.eval()
    sam_model.train() if training else sam_model.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        
        batch = utils.to_cuda(batch)
        
        b, s = batch['support_imgs'].shape[:2]
        batch['support_imgs'] = batch['support_imgs'].view(b*s,*batch['support_imgs'].shape[2:])
        batch['support_masks'] = batch['support_masks'].view(b*s,*batch['support_masks'].shape[2:])

        protos, _ = model(args.condition, batch['query_img'].expand(s,*batch['query_img'].shape).reshape(b*s,*batch['query_img'].shape[1:]), batch['support_imgs'], batch['support_masks'], training)
        protos = protos.reshape(b,s*protos.shape[1], *protos.shape[2:])

        low_masks, pred_mask = sam_model(batch['query_img'], batch['query_name'], protos)
        logit_mask = low_masks
        
        pred_mask = torch.sigmoid(logit_mask) > 0.5
        pred_mask = pred_mask.float()

        loss = model.module.compute_objective(logit_mask, batch['query_mask'])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou, _ = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Visual Prompt Encoder Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='/data/databases')
    parser.add_argument('--benchmark', type=str, default='maps', choices=['pascal', 'coco', 'maps', 'maps_siegfried'])
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--train_prompt', action='store_true')
    parser.add_argument('--train_mask', action='store_true')
    parser.add_argument('--rank', type=int, default=4) # batch size = num_gpu * bsz default num_gpu = 4
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=2) # batch size = num_gpu * bsz default num_gpu = 4
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--condition', type=str, default='mask', choices=['point', 'scribble', 'box', 'mask'])
    parser.add_argument('--use_ignore', type=bool, default=True, help='Boundaries are not considered during pascal training')
    parser.add_argument('--local-rank', type=int, default=-1, help='number of cpu threads to use during batch generation')
    parser.add_argument('--num_query', type=int, default=50)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    args = parser.parse_args()

    if args.benchmark == 'maps_siegfried':
        assert args.eval

    # Distributed setting
    local_rank = args.local_rank
    dist.init_process_group(backend='nccl')
    print('local_rank: ', local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    
    if utils.is_main_process():
        Logger.initialize(args, training=True)
    utils.fix_randseed(args.seed)

    # Model initialization
    model = VRP_encoder(args, args.backbone, False)
    sam_model = SAM_pred(args)
    if args.eval:
        model_path = 'logs/' + args.logpath + '.log/best_model.pt'
        state_dict = torch.load(model_path, map_location=device, mmap=True)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        dora_path = 'logs/' + args.logpath + '.log/model_dora.pt'
        if os.path.exists(dora_path):
            sam_model.dora.load_dora_parameters(dora_path)
            print('DORA weights found at {}'.format(dora_path))
        print('Model loaded succesfully from {}'.format(model_path))
        

    if utils.is_main_process():
        Logger.log_params(model)

    sam_model.to(device)
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    sam_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(sam_model)
    # Device setup
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    sam_model = torch.nn.parallel.DistributedDataParallel(sam_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    for param in model.module.layer0.parameters():
        param.requires_grad = False
    for param in model.module.layer1.parameters():
        param.requires_grad = False
    for param in model.module.layer2.parameters():
        param.requires_grad = False
    for param in model.module.layer3.parameters():
        param.requires_grad = False
    for param in model.module.layer4.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW([
        {'params': model.module.transformer_decoder.parameters(), "lr": args.lr},
        {'params': model.module.downsample_query.parameters(), "lr": args.lr},
        {'params': model.module.merge_1.parameters(), "lr": args.lr},
        {'params': list(filter(lambda p: p.requires_grad, sam_model.parameters())), "lr": args.lr} # DoRA parameters
        ],lr = args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    Evaluator.initialize(args)

    # Dataset initialization
    FSSDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn', shot=1)

    dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val', shot=1)

    print('Model is evaluated on {}'.format(args.benchmark))

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= args.epochs * len(dataloader_trn))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, threshold=1e-1, threshold_mode='abs', verbose=True)
    # Training 
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.epochs):

        if not args.eval:
            trn_loss, trn_miou, trn_fb_iou = train(args, epoch, model, sam_model, dataloader_trn, optimizer, scheduler, training=True)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(args, epoch, model, sam_model, dataloader_val, optimizer, scheduler, training=False)
            scheduler.step(val_miou)

        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            if utils.is_main_process():
                if not args.eval:
                    Logger.save_model_miou(model, epoch, val_miou)
                    sam_model.module.dora.save_dora_parameters('logs/' + args.logpath + '.log/model_dora.pt')

        if utils.is_main_process() and not args.eval:
            Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
            Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
            Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
            Logger.tbd_writer.flush()

        if args.eval:
            break
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')