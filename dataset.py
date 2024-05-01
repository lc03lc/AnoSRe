from PIL import Image

from torch.utils.data import Dataset
import torch

from torchvision.io import read_image
from torchvision import transforms
import torchvision
from transformers import AutoTokenizer, AutoProcessor


class CustomDataset(Dataset):
    def __init__(self, size, description, label, file_path, distribution,
                 score, max_length, tokenizer_en="roberta_base",
                 img_clip_processor="vision_clip", t_clip_tokenizer="text_clip"):
        tokenizer_dic = {
            "roberta_base": AutoTokenizer.from_pretrained("roberta-base"),
            "vision_clip": AutoProcessor.from_pretrained("openai/clip-vit-base-patch32"),
            "text_clip": AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        }

        self.size = size
        self.text_list = description
        self.label_list = label
        self.img_path_list = file_path
        self.vec_lists = distribution
        self.score_list = score
        self.max_length = max_length
        self.tokenizer = tokenizer_dic[tokenizer_en]
        self.t_clip_tokenizer = tokenizer_dic[t_clip_tokenizer]
        self.img_clip_processor = tokenizer_dic[img_clip_processor]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        one_path = "UCF-Crime/" + self.img_path_list[idx]
        img_tensor = read_image(one_path, mode=torchvision.io.ImageReadMode.RGB)
        img_tensor = transforms.Resize((224, 224))(img_tensor).float()

        image = Image.open(one_path).convert("RGB")
        one_clip_img = self.img_clip_processor(images=image, return_tensors="pt").pixel_values[0]

        one_text = self.text_list[idx]
        one_text_array = \
            self.tokenizer(one_text, return_tensors="pt", max_length=self.max_length, truncation=True,
                           padding='max_length')['input_ids'][0]
        one_clip_text = \
            self.t_clip_tokenizer(one_text, return_tensors="pt", max_length=self.max_length, truncation=True,
                                  padding="max_length")['input_ids'][0]

        one_vec = self.vec_lists[idx]
        one_vec_tensor = torch.tensor(one_vec).float()

        one_score = self.score_list[idx]
        one_score_tensor = torch.tensor(one_score).float()

        one_label = self.label_list[idx]
        one_label = torch.tensor(one_label).long()
        return img_tensor, one_clip_img, one_text_array, one_clip_text, one_vec_tensor, one_score_tensor, one_label
