import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from dataset.datautils import build_dataset
from dataset.cls_to_names import *
from dataset.imagnet_prompts import imagenet_classes
from dataset.imagenet_variants import imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from dataset.prompts_hand_craft import hand_craft_prompts_template
from dataset.transform_lib import Augmenter as Augmenter_lib
import glob


IMAGE_SIZE = 224
SHUFFLE = True


def get_classname_and_template(dset):
    if dset == 'ImageNetLT':
        classnames = imagenet_classes
        template = hand_craft_prompts_template['I']
    elif len(dset) > 1:
        classnames = eval("{}_classes".format(dset.lower()))
        template = hand_craft_prompts_template[dset.lower()]
    else:
        assert dset in ['A', 'R', 'K', 'V', 'I']
        classnames_all = imagenet_classes
        classnames = []
        if dset in ['A', 'R', 'V']:
            label_mask = eval("imagenet_{}_mask".format(dset.lower()))
            if dset == 'R':
                for i, m in enumerate(label_mask):
                    if m:
                        classnames.append(classnames_all[i])
            else:
                classnames = [classnames_all[i] for i in label_mask]
        else:
            classnames = classnames_all
        template = hand_craft_prompts_template[dset]
    return classnames, [template]


class AdvDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = glob.glob(os.path.join(data_dir, '*.png'))
        self.samples.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
        self.targets = torch.tensor([int(os.path.basename(x).split('_')[1]) for x in self.samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        target = self.targets[idx]
        image_path = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, target


def get_dataloader(args):
    transform = Augmenter_lib(
        base_transform=transforms.Compose([
            transforms.Resize(IMAGE_SIZE, interpolation=BICUBIC),
            transforms.CenterCrop(IMAGE_SIZE)
        ]),
        preprocess=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    if args.adv_dir == 'clean':
        val_dataset = build_dataset(
            args.test_sets,
            transform,
            args.data_root,
            split=getattr(args, 'lt_split', 'all'),
        )
    else:
        path = os.path.join(args.adv_dir, args.test_sets)
        assert os.path.exists(path)
        val_dataset = AdvDataset(data_dir=path, transform=transform)

    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=args.num_workers, shuffle=SHUFFLE)
    print(f"DATASET info: {args.test_sets}, {args.adv_dir}, {len(val_loader)}")
    return val_loader
