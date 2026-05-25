import argparse
import csv
import os
import time

import numpy as np
import torch

from clip import CLIP_Classifier
from utils.tools import set_random_seed
from dataset.adv_dataset import get_classname_and_template, get_dataloader

USE_GPT_PROMPTS = True
DEVICE = 'cuda'


def _normalize(vec: torch.Tensor, eps=1e-8):
    return vec / (vec.norm(dim=-1, keepdim=True) + eps)


def _entropy(logits: torch.Tensor):
    p = logits.softmax(-1)
    log_p = logits.log_softmax(-1)
    return -(p * log_p).sum(-1)


def _write_classwise_csv(path, classnames, stats):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        fields = ['class_id', 'class_name', 'test_count', 'vanilla_correct', 'dbd_correct', 'vanilla_acc', 'dbd_acc', 'db_score_mean', 'db_score_std', 'high_db_ratio']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for class_id, class_name in enumerate(classnames):
            total = int(stats['total'][class_id])
            if total > 0:
                vanilla_acc = float(stats['vanilla_correct'][class_id]) / total
                dbd_acc = float(stats['dbd_correct'][class_id]) / total
                db_mean = float(stats['db_score_sum'][class_id]) / total
                db_sq_mean = float(stats['db_score_sq_sum'][class_id]) / total
                db_std = max(db_sq_mean - db_mean * db_mean, 0.0) ** 0.5
                high_ratio = float(stats['high_db_count'][class_id]) / total
            else:
                vanilla_acc = dbd_acc = db_mean = db_std = high_ratio = 0.0
            writer.writerow({
                'class_id': class_id,
                'class_name': class_name,
                'test_count': total,
                'vanilla_correct': int(stats['vanilla_correct'][class_id]),
                'dbd_correct': int(stats['dbd_correct'][class_id]),
                'vanilla_acc': vanilla_acc,
                'dbd_acc': dbd_acc,
                'db_score_mean': db_mean,
                'db_score_std': db_std,
                'high_db_ratio': high_ratio,
            })


def infer(val_loader, model, args, classnames, thres=0.8, alpha=2.5, is_filter=True):
    n = len(val_loader)
    num_classes = len(classnames)
    stats = {
        'total': np.zeros(num_classes, dtype=np.int64),
        'vanilla_correct': np.zeros(num_classes, dtype=np.int64),
        'dbd_correct': np.zeros(num_classes, dtype=np.int64),
        'db_score_sum': np.zeros(num_classes, dtype=np.float64),
        'db_score_sq_sum': np.zeros(num_classes, dtype=np.float64),
        'high_db_count': np.zeros(num_classes, dtype=np.int64),
    }

    t0 = time.time()
    acc = 0
    vanilla_acc = 0
    for i, batch in enumerate(val_loader):
        images, target = batch[:2]
        target = target.to(DEVICE)
        images = images[0].to(DEVICE)
        target_int = int(target.squeeze().item())

        with torch.no_grad():
            logits, feats, _, text_feats, logit_scale = model.custom_inference(images)
            ent = _entropy(logits)
            _, idx = torch.topk(-ent[1:], k=16, dim=-1)

            ref_feats = feats[idx + 1, :] if is_filter else feats[1:, :]
            ori_feats = feats[0].unsqueeze(0)
            ref_dir = ref_feats - ori_feats
            ref_avg_dir = ref_dir.mean(0, keepdim=True)

            sim = _normalize(ref_dir) @ _normalize(ref_avg_dir).T
            db_score = sim.mean()
            db_score_value = float(db_score.item())
            factor = alpha if db_score > thres else 1.0

            final_feats = ref_avg_dir * factor + ori_feats
            final_logits = logit_scale * _normalize(final_feats) @ text_feats.T
            final_preds = final_logits.squeeze().argmax(dim=-1, keepdim=True)
            dbd_correct = int(final_preds.squeeze().item() == target_int)
            acc += dbd_correct

            vanilla_logits = logit_scale * _normalize(ori_feats) @ text_feats.T
            vanilla_preds = vanilla_logits.squeeze().argmax(dim=-1, keepdim=True)
            vanilla_correct = int(vanilla_preds.squeeze().item() == target_int)
            vanilla_acc += vanilla_correct

            stats['total'][target_int] += 1
            stats['vanilla_correct'][target_int] += vanilla_correct
            stats['dbd_correct'][target_int] += dbd_correct
            stats['db_score_sum'][target_int] += db_score_value
            stats['db_score_sq_sum'][target_int] += db_score_value * db_score_value
            stats['high_db_count'][target_int] += int(db_score_value > thres)

        if i % 100 == 0:
            print(f'{args.test_sets} {i}: Vanilla Acc {vanilla_acc / (i+1)}, DBD Acc {acc / (i+1)}')

    print(f'{args.test_sets} {i}:  Vanilla Acc {vanilla_acc / n}, DBD Acc {acc / n}')
    if args.classwise_csv:
        _write_classwise_csv(args.classwise_csv, classnames, stats)
        print(f'Class-wise results saved to {args.classwise_csv}')

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
    infer(val_loader, model, args, classnames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_sets', type=str, default='Caltech101')
    parser.add_argument('--adv_dir', type=str, default='clean')
    parser.add_argument('--data_root', type=str, default='./test_data')
    parser.add_argument('--lt_split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--classwise_csv', type=str, default='')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('-n', '--num_workers', default=4, type=int, metavar='N')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.adv_dir != 'clean':
        assert args.arch.replace('/', '-') in args.adv_dir
    print(args)
    main(args)
