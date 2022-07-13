import os
import sys
import time
import json
import pprint
import random
import numpy as np
import pickle
from easydict import EasyDict as EDict
from tqdm import tqdm, trange
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import h5py
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from method.config import BaseOptions
from method.model import MS_SL_Net
from method.data_provider import Dataset4MS_SL,VisDataSet4MS_SL,\
    TxtDataSet4MS_SL,collate_train,read_video_ids

from method.eval import eval_epoch,start_inference
from method.optimization import BertAdam
from utils.basic_utils import AverageMeter, BigFile, read_dict, log_config
from utils.model_utils import count_parameters


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, train_loader, optimizer, opt, epoch_i, training=True):
    logger.info("use train_epoch func for training: {}".format(training))
    model.train(mode=training)
    if opt.hard_negative_start_epoch != -1 and epoch_i >= opt.hard_negative_start_epoch:
        model.set_hard_negative(True, opt.hard_pool_size)


    # init meters
    dataloading_time = AverageMeter()
    prepare_inputs_time = AverageMeter()
    model_forward_time = AverageMeter()
    model_backward_time = AverageMeter()
    loss_meters = OrderedDict(clip_nce_loss=AverageMeter(), clip_trip_loss=AverageMeter(),
                      frame_nce_loss=AverageMeter(), frame_trip_loss=AverageMeter(),
                              loss_overall=AverageMeter())

    num_training_examples = len(train_loader)

    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader), desc="Training Iteration", total=num_training_examples):
        global_step = epoch_i * num_training_examples + batch_idx
        dataloading_time.update(time.time() - timer_dataloading)

        # continue
        timer_start = time.time()
        for k in batch.keys():
            if k != 'text_labels':
                batch[k] = batch[k].to(opt.device)

        # model_inputs = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
        prepare_inputs_time.update(time.time() - timer_start)
        timer_start = time.time()
        loss, loss_dict = model(batch['clip_video_features'],batch['frame_video_features'],batch['videos_mask'],
                                batch['text_feat'],batch['text_mask'],batch['text_labels'])
        model_forward_time.update(time.time() - timer_start)
        timer_start = time.time()
        if training:
            optimizer.zero_grad()
            loss.backward()
            if opt.grad_clip != -1:
                nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()
            model_backward_time.update(time.time() - timer_start)

            opt.writer.add_scalar("Train/LR", float(optimizer.param_groups[0]["lr"]), global_step)
            for k, v in loss_dict.items():
                opt.writer.add_scalar("Train/{}".format(k), v, global_step)

        for k, v in loss_dict.items():
            loss_meters[k].update(float(v))

        timer_dataloading = time.time()
        if opt.debug and batch_idx == 3:
            break

    if training:
        to_write = opt.train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H_%M_%S"), epoch=epoch_i,
                                                      loss_str=" ".join(["{} {:.4f}".format(k, v.avg)
                                                                         for k, v in loss_meters.items()]))
        with open(opt.train_log_filepath, "a") as f:
            f.write(to_write)
        print("Epoch time stats:")
        print("dataloading_time: max {dataloading_time.max} min {dataloading_time.min} avg {dataloading_time.avg}\n"
              "prepare_inputs_time: max {prepare_inputs_time.max} "
              "min {prepare_inputs_time.min} avg {prepare_inputs_time.avg}\n"
              "model_forward_time: max {model_forward_time.max} "
              "min {model_forward_time.min} avg {model_forward_time.avg}\n"
              "model_backward_time: max {model_backward_time.max} "
              "min {model_backward_time.min} avg {model_backward_time.avg}\n".format(
            dataloading_time=dataloading_time, prepare_inputs_time=prepare_inputs_time,
            model_forward_time=model_forward_time, model_backward_time=model_backward_time))
    else:
        for k, v in loss_meters.items():
            opt.writer.add_scalar("Eval_Loss/{}".format(k), v.avg, epoch_i)


def rm_key_from_odict(odict_obj, rm_suffix):
    """remove key entry from the OrderedDict"""
    return OrderedDict([(k, v) for k, v in odict_obj.items() if rm_suffix not in k])


def train(model, train_dataset, val_video_dataset, val_text_dataset, opt):
    # Prepare optimizer
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        if len(opt.device_ids) > 1:
            logger.info("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU


    train_loader = DataLoader(dataset=train_dataset,batch_size=opt.bsz,shuffle=True,pin_memory=opt.pin_memory,
                                num_workers=opt.num_workers,collate_fn=collate_train)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    num_train_optimization_steps = len(train_loader) * opt.n_epoch
    optimizer = BertAdam(optimizer_grouped_parameters, lr=opt.lr, weight_decay=opt.wd, warmup=opt.lr_warmup_proportion,
                         t_total=num_train_optimization_steps, schedule="warmup_linear")
    prev_best_score = 0.
    es_cnt = 0
    start_epoch = -1 if opt.eval_untrained else 0


    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            with torch.autograd.detect_anomaly():
                train_epoch(model, train_loader, optimizer, opt, epoch_i, training=True)

        with torch.no_grad():
            rsum = eval_epoch(model, val_video_dataset, val_text_dataset, opt)

        stop_score = rsum
        if stop_score > prev_best_score:
            es_cnt = 0
            prev_best_score = stop_score
            checkpoint = {"model": model.state_dict(), "model_cfg": model.config, "epoch": epoch_i}
            torch.save(checkpoint, opt.ckpt_filepath)

            logger.info("The checkpoint file has been updated.")
        else:
            es_cnt += 1
            if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                with open(opt.train_log_filepath, "a") as f:
                    f.write("Early Stop at epoch {}".format(epoch_i))
                break
        if opt.debug:
            break

    opt.writer.close()


def start_training(opt):
   
    if opt.debug:  # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True


    opt.writer = SummaryWriter(opt.tensorboard_log_dir)
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Metrics] {eval_metrics_str}\n"

    rootpath = opt.root_path

    collection = opt.collection

    trainCollection = '%strain' % collection
    valCollection = '%sval' % collection


    cap_file = {'train': '%s.caption.txt' % trainCollection,
                'val': '%s.caption.txt' % valCollection,
                }



    text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat.hdf5' % collection)


    # caption
    caption_files = {x: os.path.join(rootpath, collection, 'TextData', cap_file[x])
                     for x in cap_file}
    # Load visual features
    visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', opt.visual_feature)


    visual_feats = BigFile(visual_feat_path)
    opt.visual_feat_dim = visual_feats.ndims


    video2frames = read_dict(os.path.join(rootpath, collection, 'FeatureData', opt.visual_feature, 'video2frames.txt'))


    train_dataset = Dataset4MS_SL(caption_files['train'], visual_feats, text_feat_path, opt, video2frames=video2frames)

    val_text_dataset = TxtDataSet4MS_SL(caption_files['val'], text_feat_path, opt)

    val_video_ids_list = read_video_ids(caption_files['val'])
    val_video_dataset = VisDataSet4MS_SL(visual_feats, video2frames, opt, video_ids=val_video_ids_list)


    
    model_config = EDict(
        visual_input_size=opt.visual_feat_dim,
        query_input_size=opt.q_feat_size,
        hidden_size=opt.hidden_size,  # hidden dimension
        max_ctx_l=opt.max_ctx_l,
        max_desc_l=opt.max_desc_l,
        map_size=opt.map_size,
        input_drop=opt.input_drop,
        device=opt.device_ids,
        drop=opt.drop,
        n_heads=opt.n_heads,  # self-att heads
        initializer_range=opt.initializer_range,  # for linear layer
        margin=opt.margin,  # margin for ranking loss
        use_hard_negative=False,  # reset at each epoch
        hard_pool_size=opt.hard_pool_size)
    logger.info("model_config {}".format(model_config))

    NAME_TO_MODELS = {'MS_SL_Net':MS_SL_Net}
    model = NAME_TO_MODELS[opt.model_name](model_config)
    count_parameters(model)
    
    logger.info("Start Training...")
    train(model, train_dataset, val_video_dataset, val_text_dataset, opt)
    return opt.results_dir, opt.eval_split_name, opt.eval_path, opt.debug, opt.model_name


if __name__ == '__main__':
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    log_config(opt.results_dir, 'performance')
    model_dir, eval_split_name, eval_path, debug, model_name = start_training(opt)
    if not debug:
        model_dir = model_dir.split(os.sep)[-1]

        logger.info("\n\n\nFINISHED TRAINING!!!")
        logger.info("Evaluating model in {}".format(model_dir))
        logger.info("Input args {}".format(sys.argv[1:]))
        start_inference(opt)
