import argparse
from PIL import Image
import numpy as np
import torch

from torchvision import transforms
from clip import CLIP_Classifier
from utils.tools import set_random_seed
from dataset.adv_dataset import get_classname_and_template
from dataset.datautils import build_dataset
from torch.utils.data import Dataset, DataLoader
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import os
import torchattacks


USE_GPT_PROMPTS = True
DEVICE = 'cuda'
DATA_ROOT = './test_data'
IMAGE_SIZE = 224


def generate_adversarial_images(args):
    device = torch.device('cuda')
    set_random_seed(args.seed)
    
    # 加载模型
    classnames, prompts_template = get_classname_and_template(args.test_sets)
    model = CLIP_Classifier(args.arch, device)
    model.set_prompts(args.test_sets, class_names=classnames, prompts_template=prompts_template, use_gpt3_prompts=USE_GPT_PROMPTS)
    model.eval()

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    val_dataset = build_dataset(args.test_sets, transform, args.data_root)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 定义攻击
    if args.attack == 'pgd':
        attack = torchattacks.PGD(model, eps=args.eps/255.0, alpha=args.alpha/255.0, steps=args.steps)
    elif args.attack == 'autoattack':
        attack = torchattacks.AutoAttack(model, eps=args.eps/255.0, version="standard", seed=args.seed)
    elif args.attack == 'fab':
        attack = torchattacks.AFAB(model, eps=args.eps/255.0,  multi_targeted=True, seed=args.seed)
    elif args.attack == 'square':
        attack = torchattacks.Square(model, eps=args.eps/255.0, seed=args.seed)
    elif args.attack == 'apgd-ce':
        attack = torchattacks.APGD(model, eps=args.eps/255.0, seed=args.seed, loss='ce')
    elif args.attack == 'apgd-dlr':
        attack = torchattacks.APGDT(model, eps=args.eps/255.0, seed=args.seed)
    elif args.attack == 'cw':
        attack = torchattacks.CWLinf(model, steps=args.steps, abort_early=True)
    elif args.attack == 'fgsm':
        attack = torchattacks.FGSM(model, eps=args.eps/255.0)
    else:
        raise NotImplementedError

    tensor_to_pil_img = transforms.ToPILImage()

    arch = args.arch.replace('/', '-')
    if args.disable_gt:
        save_dir = f'./adv_images/{arch}_{args.attack}_eps{args.eps}_pseudo/{args.test_sets}'
    else:
        save_dir = f'./adv_images/{arch}_{args.attack}_eps{args.eps}/{args.test_sets}'
    os.makedirs(save_dir, exist_ok=True)

    cnt = 0
    for i, (images, target_gt) in enumerate(val_loader):
        # images: bs,3,224,224  [0,1]
        # target: bs
        images = images.to(device)
        
        if args.disable_gt:
            with torch.no_grad():
                logits = model(images)
                target_use = logits.argmax(dim=-1)
        else:
            target_use = target_gt.to(device)
    
        adv_images = attack(images, target_use).cpu()
        
        bs = len(images)
        # 转为 PIL 并保存
        for j in range(bs):
            pil_img = tensor_to_pil_img(adv_images[j])
            label = target_gt[j].item()
            name = classnames[label].replace(' ', '_').replace('/', '-')
            img_name = f'{cnt:05}_{label}_{name}.png'
            save_path = os.path.join(save_dir, img_name)
            pil_img.save(save_path)
            
            if cnt % 10 == 0:
                print(args.test_sets, cnt)
            cnt += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_sets', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./test_data')
    parser.add_argument('--arch', type=str, default='RN50')
    parser.add_argument('--attack', type=str, default='pgd')
    parser.add_argument('--eps', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--disable_gt', action='store_true', default=False) # use pseudo-label
    parser.add_argument('-n', '--num_workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    args = parser.parse_args()
    
    generate_adversarial_images(args)
    

    

if __name__ == '__main__':
    main()
    # python generate_adv_images.py --test_sets DTD -n 16 -b 256 --arch ViT-B/32 --eps 4.0 --alpha 1.0 --steps 100 --attack pgd
