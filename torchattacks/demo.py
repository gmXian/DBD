import torch
import torch.nn as nn

import torchvision
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import json
import os
import time

from .utils import get_attack

ATTACKS = [
    # ============================================== Linf
    "FGSM",
    "BIM",
    "RFGSM",
    "PGD",
    # "ESPGD",  # not work
    # "EOTPGD", # =PGD
    # "FFGSM",    # <FGSM
    # "TPGD",     # <FGSM
    "MIFGSM",   
    "UPGD",
    # "DIFGSM",  weak
    # "TIFGSM", # bug
    "Jitter",
    "NIFGSM",
    # "PGDRS",  # not work
    # "SINIFGSM", # <FGSM
    "VMIFGSM",     # slow 5, =PGD
    "VNIFGSM",     # slow 5, >FGSM, <PGD
    # "SPSA",   # not work
    "PIFGSM",   # >FGSM, <PGD
    "PIFGSMPP", # >FGSM, <PGD
    "CWLinf", 
    "CWBSLinf", # slow 8
    # "FAB",    # out of memory
    # "ZeroGrad",   # bug
    # ============================================== L0
    # "JSMA",   # bug
    "CWL0",  
    "CWBSL0",   # slow 8
    # "SparseFool", # too slow
    # "OnePixel", # not work
    # "Pixle",  # too weak
    # ============================================== L1
    "EADL1",  # slow 12
    "EADEN",  # slow 12
    # "FABL1",  # OOM
    # ============================================== L2
    "CW",  # 
    "CWBS", # slow 8
    # "PGDL2", # weak
    # "DeepFool", # too slow
    # "PGDRSL2", # not work
    # "FABL2", 
    # ============================================== AutoAttack series: Linf, L2
    # "AFAB", # too slow and weak
    # "APGD", # slow and > FGSM, < PGD
    # "APGDT", # slow 9, =PGD
    "AutoAttack", # slow 5
    # "Square", # not work
    # L0, L1, L2, Linf
    # "FMN", not work
]
def get_imagenet_dataset(split='val'):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    RESIZE = 256
    CROP_SIZE = (224, 224)
    imagenet_root = '/zhdd/dataset/liuls/FAP/imagenet/'

    transform = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    print(f"Used normalization: mean={MEAN}, std={STD}, resize={RESIZE}, crop_size={CROP_SIZE}")

    # https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
    _folder_2_class = json.load(open(os.path.join(imagenet_root, 'imagenet_class_index.json')))   # {0:["n01440764", "tench"]}
    
    folder_2_class = {}
    for m in _folder_2_class.values():
        folder_2_class[m[0]] = m[1]
    

    dataset = datasets.ImageFolder(root=os.path.join(imagenet_root, 'images', split), transform=transform)
    classnames = [folder_2_class[x] for x in dataset.classes]
    dataset.classes = classnames
    
    return dataset

@torch.no_grad()
def get_accuracy(model, data_loader, atk=None, n_limit=1e10, device=None):
    model = model.eval()

    if device is None:
        device = next(model.parameters()).device

    correct = 0
    total = 0

    for images, labels in data_loader:

        X = images.to(device)
        Y = labels.to(device)

        if atk:
            X = atk(X, Y)

        pre = model(X)

        _, pre = torch.max(pre.data, 1)
        total += pre.size(0)
        correct += (pre == Y).sum()

        if total > n_limit:
            break

    return (100 * float(correct) / total)


ds = get_imagenet_dataset('val')
dl = DataLoader(
    dataset=ds,
    batch_size=20,
    shuffle=False,
)

images, labels = iter(dl).next()
print(images.shape, labels.shape)


device = "cuda"
model = models.resnet50(pretrained=True).to(device).eval()
images = images.to(device)
labels = labels.to(device)
logits = model(images)
preds = logits.argmax(dim=-1)



# print(preds)
# print(labels)
acc = (preds == labels).sum()
print('clean acc: ', acc.item() / len(labels))

for attack_method in ATTACKS:
    # if attack_method in ['TIFGSM', 'FAB', 'ZeroGrad', 'JSMA', 'SparseFool', 'FABL1', 'DeepFool', 'FABL2']:
        # continue
    # print(attack_method)
    t0 = time.time()
    atk = get_attack(model=model, attack_method=attack_method, eps=1/255, alpha=1/255, steps=2)
    atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    adv_images = atk(images, labels)
    adv_logits = model(adv_images)
    adv_preds = adv_logits.argmax(dim=-1)

    # print(adv_preds)
    adv_acc = (adv_preds == labels).sum()
    print(f"{attack_method}: adv_acc {adv_acc.item() / len(labels)}, time: {time.time() - t0:.4f}")
    # print(attack_method, 'adv_acc: ', adv_acc.item() / len(labels), 'time: ', time.time() - t0)
    

