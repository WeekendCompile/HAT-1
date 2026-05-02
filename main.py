import os
import json
import torch
import torchvision
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import opts_egtea as opts

import time
import h5py
from tqdm import tqdm
from iou_utils import *
from eval import evaluation_detection
from tensorboardX import SummaryWriter
from dataset import VideoDataSet
from models import MYNET, SuppressNet
from loss_func import cls_loss_func,  regress_loss_func
from loss_func import MultiCrossEntropyLoss
from functools import *


def get_device_ids():
    """
    Automatically detect available GPUs and return device IDs.
    Returns a list of GPU device IDs or None if no GPU is available.
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device_ids = list(range(num_gpus))
        print(f"Found {num_gpus} GPU(s): {device_ids}")
        return device_ids
    else:
        print("No GPU available, using CPU")
        return None


def setup_model(model, device_ids):
    """
    Setup model with appropriate device configuration.
    
    Args:
        model: The model to setup
        device_ids: List of GPU device IDs or None for CPU
    
    Returns:
        Configured model
    """
    if device_ids is not None and len(device_ids) > 0:
        model = model.cuda()
        if len(device_ids) > 1:
            print(f"Using DataParallel with GPUs: {device_ids}")
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        else:
            print(f"Using single GPU: {device_ids[0]}")
    else:
        print("Using CPU")
    
    return model


def train_one_epoch(opt, model, train_dataset, optimizer, device_ids, warmup=False):
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=opt['batch_size'], shuffle=True,
                                                num_workers=0, pin_memory=True, drop_last=False)
    epoch_cost = 0
    epoch_cost_cls = 0
    epoch_cost_reg = 0
    epoch_cost_snip = 0
    epoch_cost_ctx = 0

    total_iter = len(train_dataset) // opt['batch_size']

    for n_iter, (input_data, cls_label, reg_label, snip_label) in enumerate(tqdm(train_loader)):

        if warmup:
            for g in optimizer.param_groups:
                g['lr'] = n_iter * (opt['lr']) / total_iter

        # Move data to appropriate device
        if device_ids is not None:
            input_data = input_data.float().cuda()
        else:
            input_data = input_data.float()

        # HAT+ returns 4 values: anchor cls, anchor reg, history snip cls, context cls
        act_cls, act_reg, snip_cls, ctx_cls = model(input_data)

        cost_reg = 0
        cost_cls = 0

        loss = cls_loss_func(cls_label, act_cls, use_focal=True)
        cost_cls = loss
        epoch_cost_cls += cost_cls.item()  # FIXED: was .detach().cpu().numpy()

        loss = regress_loss_func(reg_label, act_reg)
        cost_reg = loss
        epoch_cost_reg += cost_reg.item()  # FIXED: was .detach().cpu().numpy()

        loss = cls_loss_func(snip_label, snip_cls, use_focal=True)
        cost_snip = loss
        epoch_cost_snip += cost_snip.item()  # FIXED: was .detach().cpu().numpy()

        # Context supervision: same snip_label, same focal loss — direct gradient
        # to HierarchicalContextEncoder so it actually learns current activity
        loss = cls_loss_func(snip_label, ctx_cls, use_focal=True)
        cost_ctx = loss
        epoch_cost_ctx += cost_ctx.item()  # FIXED: was .detach().cpu().numpy()

        cost = (opt['alpha'] * cost_cls + opt['beta'] * cost_reg
                + opt['gamma'] * cost_snip + opt['delta'] * cost_ctx)

        epoch_cost += cost.item()  # FIXED: was .detach().cpu().numpy()

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    return n_iter, epoch_cost, epoch_cost_cls, epoch_cost_reg, epoch_cost_snip, epoch_cost_ctx


def eval_one_epoch(opt, model, test_dataset, device_ids):
    cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames = eval_frame(
        opt, model, test_dataset, device_ids
    )

    result_dict = eval_map_nms(opt, test_dataset, output_cls, output_reg, labels_cls, labels_reg)
    output_dict = {"version": "VERSION 1.3", "results": result_dict, "external_data": {}}
    outfile = open(opt["result_file"].format(opt['exp']), "w")
    json.dump(output_dict, outfile, indent=2)
    outfile.close()

    IoUmAP = evaluation_detection(opt, verbose=False)
    IoUmAP_5 = sum(IoUmAP[0:]) / len(IoUmAP[0:])

    return cls_loss, reg_loss, tot_loss, IoUmAP_5


def train(opt):
    writer = SummaryWriter()

    # Detect available GPUs
    device_ids = get_device_ids()

    # Build model and setup with available devices
    model = MYNET(opt)
    model = setup_model(model, device_ids)

    # HAT+ LR strategy (critical fix):
    # Only history_compressor + history_token get the slow 1e-6 LR —
    # these carry HAT's pre-learned history knowledge and mirror HAT's
    # original differential-LR intent.
    # All newly added parameters (context_encoder, memory_fusion,
    # mem_gate, anchor_refinement_block, etc.) train
    # at full opt['lr'] so they can actually learn from scratch.
    
    # Handle DataParallel wrapper when accessing named_parameters
    model_params = model.module.named_parameters() if isinstance(model, torch.nn.DataParallel) else model.named_parameters()
    
    slow_lr_params = [param for name, param in model_params
                      if 'history_compressor' in name
                      or 'history_token' in name]
    
    # Re-iterate for full_lr_params (named_parameters() is a generator)
    model_params = model.module.named_parameters() if isinstance(model, torch.nn.DataParallel) else model.named_parameters()
    full_lr_params = [param for name, param in model_params
                      if not ('history_compressor' in name
                              or 'history_token' in name)]

    optimizer = optim.Adam(
        [{'params': slow_lr_params, 'lr': 1e-6},
         {'params': full_lr_params}],
        lr=opt["lr"],
        weight_decay=opt["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["lr_step"])

    train_dataset = VideoDataSet(opt, subset="train")
    test_dataset = VideoDataSet(opt, subset=opt['inference_subset'])

    warmup = False

    for n_epoch in range(opt['epoch']):
        if n_epoch >= 1:
            warmup = False

        n_iter, epoch_cost, epoch_cost_cls, epoch_cost_reg, epoch_cost_snip, epoch_cost_ctx = train_one_epoch(
            opt, model, train_dataset, optimizer, device_ids, warmup
        )

        writer.add_scalars('data/cost', {'train': epoch_cost / (n_iter + 1)}, n_epoch)
        print("training loss(epoch %d): %.03f, cls - %f, reg - %f, snip - %f, ctx - %f, lr - %f" % (
            n_epoch,
            epoch_cost / (n_iter + 1),
            epoch_cost_cls / (n_iter + 1),
            epoch_cost_reg / (n_iter + 1),
            epoch_cost_snip / (n_iter + 1),
            epoch_cost_ctx / (n_iter + 1),
            optimizer.param_groups[-1]["lr"])
        )

        scheduler.step()
        model.eval()

        cls_loss, reg_loss, tot_loss, IoUmAP_5 = eval_one_epoch(opt, model, test_dataset, device_ids)

        writer.add_scalars('data/mAP', {'test': IoUmAP_5}, n_epoch)
        print("testing loss(epoch %d): %.03f, cls - %f, reg - %f, mAP Avg - %f" % (
            n_epoch, tot_loss, cls_loss, reg_loss, IoUmAP_5
        ))

        # Use .module to access state_dict and custom attributes when using DataParallel
        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
        state = {'epoch': n_epoch + 1,
                 'state_dict': model_to_save.state_dict()}
        torch.save(state, opt["checkpoint_path"] + "/" + opt["exp"] + "_checkpoint_" + str(n_epoch + 1) + ".pth.tar")

        if IoUmAP_5 > model_to_save.best_map:
            model_to_save.best_map = IoUmAP_5
            torch.save(state, opt["checkpoint_path"] + "/" + opt["exp"] + "_ckp_best.pth.tar")

        model.train()

    writer.close()
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    return model_to_save.best_map


def eval_frame(opt, model, dataset, device_ids):
    test_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=opt['batch_size'], shuffle=False,
                                               num_workers=0, pin_memory=True, drop_last=False)

    labels_cls = {}
    labels_reg = {}
    output_cls = {}
    output_reg = {}
    for video_name in dataset.video_list:
        labels_cls[video_name] = []
        labels_reg[video_name] = []
        output_cls[video_name] = []
        output_reg[video_name] = []

    start_time = time.time()
    total_frames = 0
    epoch_cost = 0
    epoch_cost_cls = 0
    epoch_cost_reg = 0

    for n_iter, (input_data, cls_label, reg_label, _) in enumerate(tqdm(test_loader)):
        # Move data to appropriate device
        if device_ids is not None:
            input_data = input_data.float().cuda()
        else:
            input_data = input_data.float()
            
        act_cls, act_reg, _, _ = model(input_data)
        cost_reg = 0
        cost_cls = 0

        loss = cls_loss_func(cls_label, act_cls)
        cost_cls = loss

        epoch_cost_cls += cost_cls.item()  # FIXED: was .detach().cpu().numpy()

        loss = regress_loss_func(reg_label, act_reg)
        cost_reg = loss
        epoch_cost_reg += cost_reg.item()  # FIXED: was .detach().cpu().numpy()

        cost = opt['alpha'] * cost_cls + opt['beta'] * cost_reg

        epoch_cost += cost.item()  # FIXED: was .detach().cpu().numpy()

        act_cls = torch.softmax(act_cls, dim=-1)

        total_frames += input_data.size(0)

        for b in range(0, input_data.size(0)):
            video_name, st, ed, data_idx = dataset.inputs[n_iter * opt['batch_size'] + b]
            output_cls[video_name] += [act_cls[b, :].detach().cpu().numpy()]
            output_reg[video_name] += [act_reg[b, :].detach().cpu().numpy()]
            labels_cls[video_name] += [cls_label[b, :].numpy()]
            labels_reg[video_name] += [reg_label[b, :].numpy()]

    end_time = time.time()
    working_time = end_time - start_time

    for video_name in dataset.video_list:
        labels_cls[video_name] = np.stack(labels_cls[video_name], axis=0)
        labels_reg[video_name] = np.stack(labels_reg[video_name], axis=0)
        output_cls[video_name] = np.stack(output_cls[video_name], axis=0)
        output_reg[video_name] = np.stack(output_reg[video_name], axis=0)

    cls_loss = float(epoch_cost_cls / n_iter)
    reg_loss = float(epoch_cost_reg / n_iter)
    tot_loss = float(epoch_cost / n_iter)

    return cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames


def eval_map_nms(opt, dataset, output_cls, output_reg, labels_cls, labels_reg):
    result_dict = {}
    proposal_dict = []

    num_class = opt["num_of_class"]
    unit_size = opt['segment_size']
    threshold = opt['threshold']
    anchors = opt['anchors']

    for video_name in dataset.video_list:
        duration = dataset.video_len[video_name]
        video_time = float(dataset.video_dict[video_name]["duration"])
        frame_to_time = 100.0 * video_time / duration

        for idx in range(0, duration):
            cls_anc = output_cls[video_name][idx]
            reg_anc = output_reg[video_name][idx]

            proposal_anc_dict = []
            for anc_idx in range(0, len(anchors)):
                cls = np.argwhere(cls_anc[anc_idx][:-1] > opt['threshold']).reshape(-1)

                if len(cls) == 0:
                    continue

                ed = idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                length = anchors[anc_idx] * np.exp(reg_anc[anc_idx][1])
                st = ed - length

                for cidx in range(0, len(cls)):
                    label = cls[cidx]
                    tmp_dict = {}
                    tmp_dict["segment"] = [float(st * frame_to_time / 100.0), float(ed * frame_to_time / 100.0)]
                    tmp_dict["score"] = float(cls_anc[anc_idx][label])
                    tmp_dict["label"] = dataset.label_name[label]
                    tmp_dict["gentime"] = float(idx * frame_to_time / 100.0)
                    proposal_anc_dict.append(tmp_dict)

            proposal_dict += proposal_anc_dict

        proposal_dict = non_max_suppression(proposal_dict, overlapThresh=opt['soft_nms'])

        result_dict[video_name] = proposal_dict
        proposal_dict = []

    return result_dict


def eval_map_supnet(opt, dataset, output_cls, output_reg, labels_cls, labels_reg, device_ids):
    # Build SuppressNet and setup with available devices
    model = SuppressNet(opt)
    model = setup_model(model, device_ids)
    
    checkpoint = torch.load(opt["checkpoint_path"] + "/ckp_best_suppress.pth.tar")
    base_dict = checkpoint['state_dict']
    model_to_load = model.module if isinstance(model, torch.nn.DataParallel) else model
    model_to_load.load_state_dict(base_dict)
    model.eval()

    result_dict = {}
    proposal_dict = []

    num_class = opt["num_of_class"]
    unit_size = opt['segment_size']
    threshold = opt['threshold']
    anchors = opt['anchors']

    for video_name in dataset.video_list:
        duration = dataset.video_len[video_name]
        video_time = float(dataset.video_dict[video_name]["duration"])
        frame_to_time = 100.0 * video_time / duration
        conf_queue = torch.zeros((unit_size, num_class - 1))

        for idx in range(0, duration):
            cls_anc = output_cls[video_name][idx]
            reg_anc = output_reg[video_name][idx]

            proposal_anc_dict = []
            for anc_idx in range(0, len(anchors)):
                cls = np.argwhere(cls_anc[anc_idx][:-1] > opt['threshold']).reshape(-1)

                if len(cls) == 0:
                    continue

                ed = idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                length = anchors[anc_idx] * np.exp(reg_anc[anc_idx][1])
                st = ed - length

                for cidx in range(0, len(cls)):
                    label = cls[cidx]
                    tmp_dict = {}
                    tmp_dict["segment"] = [float(st * frame_to_time / 100.0), float(ed * frame_to_time / 100.0)]
                    tmp_dict["score"] = float(cls_anc[anc_idx][label])
                    tmp_dict["label"] = dataset.label_name[label]
                    tmp_dict["gentime"] = float(idx * frame_to_time / 100.0)
                    proposal_anc_dict.append(tmp_dict)

            proposal_anc_dict = non_max_suppression(proposal_anc_dict, overlapThresh=opt['soft_nms'])

            conf_queue[:-1, :] = conf_queue[1:, :].clone()
            conf_queue[-1, :] = 0
            for proposal in proposal_anc_dict:
                cls_idx = dataset.label_name.index(proposal['label'])
                conf_queue[-1, cls_idx] = proposal["score"]

            minput = conf_queue.unsqueeze(0)
            if device_ids is not None:
                minput = minput.cuda()
            suppress_conf = model(minput)
            suppress_conf = suppress_conf.squeeze(0).detach().cpu().numpy()

            for cls in range(0, num_class - 1):
                if suppress_conf[cls] > opt['sup_threshold']:
                    for proposal in proposal_anc_dict:
                        if proposal['label'] == dataset.label_name[cls]:
                            if check_overlap_proposal(proposal_dict, proposal, overlapThresh=opt['soft_nms']) is None:
                                proposal_dict.append(proposal)

        result_dict[video_name] = proposal_dict
        proposal_dict = []

    return result_dict


def test_frame(opt):
    # Detect available GPUs
    device_ids = get_device_ids()
    
    # Build MYNET and setup with available devices
    model = MYNET(opt)
    model = setup_model(model, device_ids)
    
    checkpoint = torch.load(opt["checkpoint_path"] + "/ckp_best.pth.tar")
    base_dict = checkpoint['state_dict']
    model_to_load = model.module if isinstance(model, torch.nn.DataParallel) else model
    model_to_load.load_state_dict(base_dict)
    model.eval()

    dataset = VideoDataSet(opt, subset=opt['inference_subset'])
    outfile = h5py.File(opt['frame_result_file'].format(opt['exp']), 'w')

    cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames = eval_frame(
        opt, model, dataset, device_ids
    )

    print("testing loss: %f, cls_loss: %f, reg_loss: %f" % (tot_loss, cls_loss, reg_loss))

    for video_name in dataset.video_list:
        o_cls = output_cls[video_name]
        o_reg = output_reg[video_name]
        l_cls = labels_cls[video_name]
        l_reg = labels_reg[video_name]

        dset_predcls = outfile.create_dataset(video_name + '/pred_cls', o_cls.shape, maxshape=o_cls.shape, chunks=True, dtype=np.float32)
        dset_predcls[:, :] = o_cls[:, :]
        dset_predreg = outfile.create_dataset(video_name + '/pred_reg', o_reg.shape, maxshape=o_reg.shape, chunks=True, dtype=np.float32)
        dset_predreg[:, :] = o_reg[:, :]
        dset_labelcls = outfile.create_dataset(video_name + '/label_cls', l_cls.shape, maxshape=l_cls.shape, chunks=True, dtype=np.float32)
        dset_labelcls[:, :] = l_cls[:, :]
        dset_labelreg = outfile.create_dataset(video_name + '/label_reg', l_reg.shape, maxshape=l_reg.shape, chunks=True, dtype=np.float32)
        dset_labelreg[:, :] = l_reg[:, :]
    outfile.close()

    print("working time : {}s, {}fps, {} frames".format(working_time, total_frames / working_time, total_frames))


def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return forward_orig(*args, **kwargs)

    m.forward = wrap


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []


def test(opt):
    # Detect available GPUs
    device_ids = get_device_ids()
    
    # Build MYNET and setup with available devices
    model = MYNET(opt)
    model = setup_model(model, device_ids)
    
    checkpoint = torch.load(opt["checkpoint_path"] + "/" + opt['exp'] + "_ckp_best.pth.tar")
    base_dict = checkpoint['state_dict']
    model_to_load = model.module if isinstance(model, torch.nn.DataParallel) else model
    model_to_load.load_state_dict(base_dict)
    model.eval()

    dataset = VideoDataSet(opt, subset=opt['inference_subset'])

    cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames = eval_frame(
        opt, model, dataset, device_ids
    )

    if opt["pptype"] == "nms":
        result_dict = eval_map_nms(opt, dataset, output_cls, output_reg, labels_cls, labels_reg)
    if opt["pptype"] == "net":
        result_dict = eval_map_supnet(opt, dataset, output_cls, output_reg, labels_cls, labels_reg, device_ids)

    output_dict = {"version": "VERSION 1.3", "results": result_dict, "external_data": {}}
    outfile = open(opt["result_file"].format(opt['exp']), "w")
    json.dump(output_dict, outfile, indent=2)
    outfile.close()

    mAP = evaluation_detection(opt)


def test_online(opt):
    # Detect available GPUs
    device_ids = get_device_ids()
    
    # Build MYNET and setup with available devices
    model = MYNET(opt)
    model = setup_model(model, device_ids)
    
    checkpoint = torch.load(opt["checkpoint_path"] + "/ckp_best.pth.tar")
    base_dict = checkpoint['state_dict']
    model_to_load = model.module if isinstance(model, torch.nn.DataParallel) else model
    model_to_load.load_state_dict(base_dict)
    model.eval()

    # Build SuppressNet and setup with available devices
    sup_model = SuppressNet(opt)
    sup_model = setup_model(sup_model, device_ids)
    
    checkpoint = torch.load(opt["checkpoint_path"] + "/ckp_best_suppress.pth.tar")
    base_dict = checkpoint['state_dict']
    sup_model_to_load = sup_model.module if isinstance(sup_model, torch.nn.DataParallel) else sup_model
    sup_model_to_load.load_state_dict(base_dict)
    sup_model.eval()

    dataset = VideoDataSet(opt, subset=opt['inference_subset'])
    test_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=1, shuffle=False,
                                               num_workers=0, pin_memory=True, drop_last=False)

    result_dict = {}
    proposal_dict = []

    num_class = opt["num_of_class"]
    unit_size = opt['segment_size']
    threshold = opt['threshold']
    anchors = opt['anchors']

    start_time = time.time()
    total_frames = 0

    for video_name in dataset.video_list:
        input_queue = torch.zeros((unit_size, opt['feat_dim']))
        sup_queue = torch.zeros(((unit_size, num_class - 1)))

        duration = dataset.video_len[video_name]
        video_time = float(dataset.video_dict[video_name]["duration"])
        frame_to_time = 100.0 * video_time / duration

        for idx in range(0, duration):
            total_frames += 1
            input_queue[:-1, :] = input_queue[1:, :].clone()
            input_queue[-1:, :] = dataset._get_base_data(video_name, idx, idx + 1)

            minput = input_queue.unsqueeze(0)
            if device_ids is not None:
                minput = minput.cuda()
            act_cls, act_reg, _, _ = model(minput)
            act_cls = torch.softmax(act_cls, dim=-1)

            cls_anc = act_cls.squeeze(0).detach().cpu().numpy()
            reg_anc = act_reg.squeeze(0).detach().cpu().numpy()

            proposal_anc_dict = []
            for anc_idx in range(0, len(anchors)):
                cls = np.argwhere(cls_anc[anc_idx][:-1] > opt['threshold']).reshape(-1)

                if len(cls) == 0:
                    continue

                ed = idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                length = anchors[anc_idx] * np.exp(reg_anc[anc_idx][1])
                st = ed - length

                for cidx in range(0, len(cls)):
                    label = cls[cidx]
                    tmp_dict = {}
                    tmp_dict["segment"] = [float(st * frame_to_time / 100.0), float(ed * frame_to_time / 100.0)]
                    tmp_dict["score"] = float(cls_anc[anc_idx][label])
                    tmp_dict["label"] = dataset.label_name[label]
                    tmp_dict["gentime"] = float(idx * frame_to_time / 100.0)
                    proposal_anc_dict.append(tmp_dict)

            proposal_anc_dict = non_max_suppression(proposal_anc_dict, overlapThresh=opt['soft_nms'])

            sup_queue[:-1, :] = sup_queue[1:, :].clone()
            sup_queue[-1, :] = 0
            for proposal in proposal_anc_dict:
                cls_idx = dataset.label_name.index(proposal['label'])
                sup_queue[-1, cls_idx] = proposal["score"]

            minput = sup_queue.unsqueeze(0)
            if device_ids is not None:
                minput = minput.cuda()
            suppress_conf = sup_model(minput)
            suppress_conf = suppress_conf.squeeze(0).detach().cpu().numpy()

            for cls in range(0, num_class - 1):
                if suppress_conf[cls] > opt['sup_threshold']:
                    for proposal in proposal_anc_dict:
                        if proposal['label'] == dataset.label_name[cls]:
                            if check_overlap_proposal(proposal_dict, proposal, overlapThresh=opt['soft_nms']) is None:
                                proposal_dict.append(proposal)

        result_dict[video_name] = proposal_dict
        proposal_dict = []

    end_time = time.time()
    working_time = end_time - start_time
    print("working time : {}s, {}fps, {} frames".format(working_time, total_frames / working_time, total_frames))

    output_dict = {"version": "VERSION 1.3", "results": result_dict, "external_data": {}}
    outfile = open(opt["result_file"].format(opt['exp']), "w")
    json.dump(output_dict, outfile, indent=2)
    outfile.close()

    evaluation_detection(opt)


def main(opt):
    max_perf = 0
    if opt['mode'] == 'train':
        max_perf = train(opt)
    if opt['mode'] == 'test':
        test(opt)
    if opt['mode'] == 'test_frame':
        test_frame(opt)
    if opt['mode'] == 'test_online':
        test_online(opt)
    if opt['mode'] == 'eval':
        evaluation_detection(opt)

    return max_perf


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    opt_file = open(opt["checkpoint_path"] + "/" + opt["exp"] + "_opts.json", "w")
    json.dump(opt, opt_file)
    opt_file.close()

    if opt['seed'] >= 0:
        seed = opt['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        # random.seed(seed)

    opt['anchors'] = [int(item) for item in opt['anchors'].split(',')]

    main(opt)
    while(opt['wterm']):
        pass
