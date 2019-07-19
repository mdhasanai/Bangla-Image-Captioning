import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision as tv
import pandas as pd

import nltk
from PIL import Image
import os
import pickle
#from vocab import Vocabulary

from config import image_dir

class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, df, vocab, transform=None,loader=tv.datasets.folder.default_loader):
        self.df = df
        self.transform = transform
        self.loader = loader
        self.token= self.df['Token']
        self.text= self.df['text']
        self.vocab = vocab
        self.root = image_dir
        

    def __getitem__(self, index):
        
        
        caption = str(self.text[index])
        img_id = self.token[index]
        vocab = self.vocab
        
        image = Image.open(os.path.join(self.root, img_id)).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption))
        captions = []
        captions.append(vocab('<start>'))
        captions.extend([vocab(token) for token in tokens])
        captions.append(vocab('<end>'))
        target = torch.Tensor(captions)
        #print(tokens)
        
        
        return image, target


    def __len__(self):
        n, _ = self.df.shape
        return n
    
#     def __len1__(self):
#         return self.text
def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader( csv_file, vocab, transform, batch_size, shuffle, num_workers):

    train_dataset = ImagesDataset(
                        csv_file,vocab,transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           collate_fn=collate_fn)

    return train_loader
