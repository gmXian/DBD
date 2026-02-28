import argparse
import time

from PIL import Image
import numpy as np
import torch

from clip import CLIP_Classifier
from utils.tools import set_random_seed
from dataset.adv_dataset import get_classname_and_template, get_dataloader

import os

USE_GPT_PROMPTS = True
DEVICE = 'cuda'


def _normalize(vec: torch.Tensor, eps=1e-8):
    return vec / (vec.norm(dim=-1, keepdim=True) + eps)

def _entropy(logits: torch.Tensor):
    p = logits.softmax(-1)
    log_p = logits.log_softmax(-1)
    ent = -(p * log_p).sum(-1)
    return ent

def infer(val_loader, model, args, thres=0.8, alpha=2.5, is_filter=True):
    n = len(val_loader)

    t0 = time.time()
    acc = 0
    vanilla_acc = 0
    for i, (images, target) in enumerate(val_loader):
        # images: bs,n,3,224,224  [0,1]
        target = target.to(DEVICE)
        images = images[0].to(DEVICE)   # 32,3,224,224
        
        with torch.no_grad():
            logits, feats, _, text_feats, logit_scale = model.custom_inference(images)
            
            ent = _entropy(logits)
            v, idx = torch.topk(-ent[1:], k=16, dim=-1)
            
            if is_filter:
                ref_feats = feats[idx+1, :]  # 31,512
            else:
                ref_feats = feats[1:, :]  # 31,512
            ori_feats = feats[0].unsqueeze(0)  # 1,512
            ref_dir = ref_feats - ori_feats
            ref_avg_dir = ref_dir.mean(0, keepdim=True)  # 1,512

            sim = _normalize(ref_dir) @ _normalize(ref_avg_dir).T   # 31, 1
            db_score = sim.mean()            
            factor = alpha if db_score > thres else 1.0

            final_feats = ref_avg_dir * factor + ori_feats
            final_logits = logit_scale * _normalize(final_feats) @ text_feats.T    # 1, c
            final_preds = final_logits.squeeze().argmax(dim=-1, keepdim=True)
            if final_preds.squeeze() == target.squeeze():
                acc += 1

            vanilla_logits = logit_scale * _normalize(ori_feats) @ text_feats.T    # 1, c
            vanilla_preds = vanilla_logits.squeeze().argmax(dim=-1, keepdim=True)
            if vanilla_preds.squeeze() == target.squeeze():
                vanilla_acc += 1

        if i % 100 == 0:
            print(f'{args.test_sets} {i}: Vanilla Acc {vanilla_acc / (i+1)}, DBD Acc {acc / (i+1)}')
    
    print(f'{args.test_sets} {i}:  Vanilla Acc {vanilla_acc / n}, DBD Acc {acc / n}')

    t1 = time.time() - t0
    print(f"All Time: {t1} s, {t1 / n} s/image")
    

def main(args):
    print('Use CuPL', USE_GPT_PROMPTS)

    set_random_seed(args.seed)

    classnames, prompts_template = get_classname_and_template(args.test_sets)
    model = CLIP_Classifier(args.arch, DEVICE)
    model.set_prompts(args.test_sets, class_names=classnames, prompts_template=prompts_template, use_gpt3_prompts=USE_GPT_PROMPTS)
    model.eval()
    val_loader = get_dataloader(args)
    infer(val_loader, model, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_sets', type=str, default='Caltech101')
    parser.add_argument('--adv_dir', type=str, default='clean')             # path of adv images
    parser.add_argument('--data_root', type=str, default='./test_data')     # path of clean images 
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')     # RN50, ViT-B/32, ViT-B/32
    parser.add_argument('-n', '--num_workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    if args.adv_dir != 'clean':
        assert args.arch.replace('/', '-') in args.adv_dir

    print(args)

    main(args)
    # python infer_dbd.py --test_sets DTD --adv_dir adv_images/ViT-B-32_pgd_eps4.0 --arch ViT-B/32 -n 8 --seed 0