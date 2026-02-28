import torch
import torch.nn.functional as F
from torchvision import transforms
from .clip import load, tokenize
import os, json, glob


DOWNLOAD_ROOT = 'ckpts/clip'
remap = {
    'DTD': 'dtd', 
    'Flower102': 'flowers102', 
    'Food101': 'food101', 
    'Cars': 'stanfordcars', 
    'SUN397': 'sun397', 
    'Aircraft': 'fgvcaircraft',
    'Pets': 'oxfordpets', 
    'Caltech101': 'caltech101', 
    'UCF101': 'ucf101', 
    'eurosat': 'eurosat'
}

class CLIP_Classifier(torch.nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        model, transform = load(model_name, device, download_root=DOWNLOAD_ROOT)
        self.model = model
        # self.transform = transform
        self.img_normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ).to(device)
        
        self.device = device
        self.dtype = self.model.dtype
        
    @torch.no_grad()
    def set_prompts(self, data_name, class_names, prompts_template, use_gpt3_prompts=True, gpt3_prompts_dir='./dataset/gpt3_prompts'):
        assert data_name in list(remap.keys()) + ['A', 'R', 'K', 'V', 'I']
        if data_name in list(remap.keys()):
            data_name = remap[data_name]
        elif data_name in ['A', 'R', 'K', 'V', 'I']:
            data_name = 'imagenet'
        else:
            raise ValueError
        
        print('Preprocess prompts...')
        path = f'{gpt3_prompts_dir}/CuPL_prompts_{data_name}.json'

        embeddings_dict = {}
        prompts_json = json.load(open(path, 'r'))
        for name, prompts in prompts_json.items():
            name = name.replace('_', ' ')
            prompts = [t.format(name) for t in prompts_template] + prompts
            prompts_token = tokenize(prompts).to(self.device)
            
            embeddings = self.model.encode_text(prompts_token) # n,c
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            
            mean_embeddings = embeddings.mean(dim=0)
            mean_embeddings /= mean_embeddings.norm()
            
            temp_embeddings = embeddings[:len(prompts_template)].mean(dim=0)
            temp_embeddings /= temp_embeddings.norm()

            embeddings_dict[name] = dict(
                prompts=prompts,    # n
                embeddings=embeddings.cpu(),    # n,1024
                temp_embeddings=temp_embeddings.cpu(),
                mean_embeddings=mean_embeddings.cpu() # 1024
            )

        if use_gpt3_prompts:     # use template and gpt3 prompts
            print("Use template and gpt3 prompts.")
            self.text_embeddings = torch.stack([embeddings_dict[name.replace('_', ' ')]['mean_embeddings'] for name in class_names]).to(self.device, self.dtype)
        else:       # use template prompts
            print("Use template prompts.")
            self.text_embeddings = torch.stack([embeddings_dict[name.replace('_', ' ')]['temp_embeddings'] for name in class_names]).to(self.device, self.dtype)
        
        self.class_names = class_names
    
    def forward(self, images):
        # images: n,3,h,w, no-normalize !!!
        images = self.img_normalize(images)
        
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self.text_embeddings
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t() # n,c

        return logits
    
    def custom_inference(self, images):
        images = self.img_normalize(images)
        
        image_features = self.model.encode_image(images)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self.text_embeddings
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_features_norm @ text_features.t() # n,c

        return logits, image_features_norm, image_features, text_features, logit_scale
    