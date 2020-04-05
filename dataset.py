import os,json
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.models as models
from collections import defaultdict
from utils import *
class VisualSemanticEmbedding(nn.Module):
    def __init__(self):
        super(VisualSemanticEmbedding, self).__init__()
        self.model = models.resnet18(pretrained=True)

    def forward(self, img):
        # image feature
        img = self.model.conv1(img)
        img = self.model.relu(self.model.bn1(img))
        img = self.model.maxpool(img)
        img = self.model.layer1(img)
        img = self.model.layer2(img)
        img = self.model.layer3(img)
        img = self.model.layer4(img)
        img = self.model.avgpool(img)
        img = img.view(img.size(0),-1)
        return img

class MultimodalDataset(Dataset):
    '''
    Preprocessed Caltech-UCSD Birds 200-2011 and Oxford 102 Category Flowers
    datasets, used in ``Learning Deep Representations of Fine-grained Visual
    Descriptions``.
    Download data from: https://github.com/reedscot/cvpr2016.
    Arguments:
        data_dir (string): path to directory containing dataset files.
        split (string): which data split to load.
    '''
    def __init__(self, image_dir, captions_json, image_size=(64,64), embed_ndim=1024, image_json=None):
        super().__init__()
        self.image_dir = image_dir
        with open(captions_json, 'r') as f:
            captions_data = json.load(f)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        if image_json == None:
            self.img_encoder = VisualSemanticEmbedding()
            self.image_size = image_size
            self.image_ids = []
            self.image = {}
            for image_data in captions_data['images']:
                image_id = image_data['id']
                filename = image_data['file_name']
                self.image_ids.append(image_id)
                image_path = os.path.join(self.image_dir, filename)
                with open(image_path, 'rb') as f:
                    with Image.open(f).resize(self.image_size,Image.BILINEAR) as image:
                        image = self.transform(image.convert('RGB'))
                img = self.img_encoder(image.unsqueeze(0))
                img = img.squeeze().detach()
                self.image[image_id] = img
            with open("./COCO.pkl", "wb") as fp:   #Pickling
                pickle.dump(self.image, fp, protocol = pickle.HIGHEST_PROTOCOL)
        else:
            self.image_ids = []
            for image_data in captions_data['images']:
                image_id = image_data['id']
                self.image_ids.append(image_id)
            with open(image_json,'rb') as f:
                self.image = pickle.load(f)
        self.image_id_to_captions = defaultdict(list)
        
        for caption_data in captions_data['annotations']:
            image_id = caption_data['image_id']
            self.image_id_to_captions[image_id].append(prepare_text(caption_data['caption']))

    def __len__(self):
        # WARNING: this number is somewhat arbitrary, since we do not
        # necessarily use all instances in an epoch
        return len(self.image_id_to_captions)


    def __getitem__(self, index):
        image_id = self.image_ids[index]
        cls_txts = self.image_id_to_captions[image_id]
        
        id_txt = torch.randint(len(cls_txts), (1,))
        
        txt = cls_txts[id_txt].squeeze()
        img = self.image[image_id].squeeze()

        return {'img': img, 'txt': txt}
