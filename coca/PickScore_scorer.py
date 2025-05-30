import os
import torch
from tqdm import tqdm
from transformers import AutoModel, CLIPProcessor
import requests


class PickScoreScorer(torch.nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        # link="https://hf-mirror.com/yuvalkirstain/PickScore_v1"
        checkpoint_path = "yuvalkirstain/PickScore_v1"
        # checkpoint_path = f"{os.path.expanduser('~')}/.cache/PickScore_v1"
        # os.makedirs(os.path.expanduser('~/.cache/PickScore_v1'), exist_ok=True)
        # checkpoint_path = f"{os.path.expanduser('~')}/.cache/PickScore_v1/"
        
        # Download the file if it doesn't exist
        # if not os.path.exists(checkpoint_path):
        #     response = requests.get(link, stream=True)
        #     total_size = int(response.headers.get('content-length', 0))

        #     with open(checkpoint_path, 'wb') as file, tqdm(
        #         desc="Downloading HPS_v2",
        #         total=total_size,
        #         unit='iB',
        #         unit_scale=True,
        #         unit_divisor=1024,
        #     ) as progress_bar:
        #         for data in response.iter_content(chunk_size=1024):
        #             size = file.write(data)
        #             progress_bar.update(size)
                    
        self.model = AutoModel.from_pretrained(checkpoint_path).eval().to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def __call__(self, images, prompts):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(self.device) for k, v in inputs.items()}
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        image_embeds = self.model.get_image_features(**inputs)
        image_embeds = image_embeds / torch.norm(image_embeds, dim=-1, keepdim=True)
        text_embeds = self.model.get_text_features(**text_inputs)
        text_embeds = text_embeds / torch.norm(text_embeds, dim=-1, keepdim=True)
        logits_per_image = image_embeds @ text_embeds.T
        scores = torch.diagonal(logits_per_image)

        return scores
