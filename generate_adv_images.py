import argparse
from PIL import Image
import torch

from torchvision import transforms
from clip import CLIP_Classifier
from utils.tools import set_random_seed
from dataset.adv_dataset import get_classname_and_template
from dataset.datautils import build_dataset
from torch.utils.data import DataLoader
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import os
import torchattacks


USE_GPT_PROMPTS = True
DEVICE = 'cuda'
IMAGE_SIZE = 224


def generate_adversarial_images(args):
    device = torch.device('cuda')
    set_random_seed(args.seed)

    classnames, prompts_template = get_classname_and_template(args.test_sets)
    model = CLIP_Classifier(args.arch, device)
    model.set_prompts(args.test_sets, class_names=classnames, prompts_template=prompts_template, use_gpt3_prompts=USE_GPT_PROMPTS)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    val_dataset = build_dataset(
        args.test_sets,
        transform,
        args.data_root,
        split=getattr(args, 'lt_split', 'all'),
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.attack == 'pgd':
        attack = torchattacks.PGD(model, eps=args.eps/255.0, alpha=args.alpha/255.0, steps=args.steps)
    elif args.attack == 'autoattack':
        attack = torchattacks.AutoAttack(model, eps=args.eps/255.0, version="standard", seed=args.seed)
    elif args.attack == 'fab':
        attack = torchattacks.AFAB(model, eps=args.eps/255.0, multi_targeted=True, seed=args.seed)
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
    suffix = f'{arch}_{args.attack}_eps{args.eps}'
    if args.disable_gt:
        suffix = f'{suffix}_pseudo'
    if args.test_sets == 'ImageNetLT':
        suffix = f'{suffix}_{args.lt_split}'
    save_dir = os.path.join(args.output_root, suffix, args.test_sets)
    os.makedirs(save_dir, exist_ok=True)

    cnt = 0
    for i, batch in enumerate(val_loader):
        images, target_gt = batch[:2]
        images = images.to(device)

        if args.disable_gt:
            with torch.no_grad():
                logits = model(images)
                target_use = logits.argmax(dim=-1)
        else:
            target_use = target_gt.to(device)

        adv_images = attack(images, target_use).cpu()

        bs = len(images)
        for j in range(bs):
            pil_img = tensor_to_pil_img(adv_images[j])
            label = int(target_gt[j].item())
            name = classnames[label].replace(' ', '_').replace('/', '-')
            img_name = f'{cnt:05}_{label}_{name}.png'
            save_path = os.path.join(save_dir, img_name)
            pil_img.save(save_path)

            if cnt % 10 == 0:
                print(args.test_sets, cnt)
            cnt += 1

    print(f'Adversarial images saved to {save_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_sets', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='./test_data')
    parser.add_argument('--lt_split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--output_root', type=str, default='./adv_images')
    parser.add_argument('--arch', type=str, default='RN50')
    parser.add_argument('--attack', type=str, default='pgd')
    parser.add_argument('--eps', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--disable_gt', action='store_true', default=False)
    parser.add_argument('-n', '--num_workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    args = parser.parse_args()

    generate_adversarial_images(args)


if __name__ == '__main__':
    main()
