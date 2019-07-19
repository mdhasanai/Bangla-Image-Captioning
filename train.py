import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import pandas as pd
from torchvision import transforms
from torch import multiprocessing
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence


#from dataloader import get_loader 
from vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
#import tqdm
from dataloader import get_loader
from config import *


def train(vocab):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([ 
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

	# Build data loader for the train
    train_df = pd.read_csv(train_file, names=['Token', 'text'],skipinitialspace=True) 
    train_loader = get_loader(train_df, vocab, 
	                         transform, batch_size,
	                         shuffle=True, num_workers=num_workers) 

	# Build data loader for validation
    valid_df = pd.read_csv(valid_file, names=['Token', 'text'],skipinitialspace=True) 
    valid_loader = get_loader(valid_df, vocab, 
	                         transform, batch_size,
	                         shuffle=True, num_workers=num_workers) 
	                       
	#image, caption,lengt = next(iter(train_loader))
	
	# Build the models
    encoder = EncoderCNN(embed_size,freeze=True).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)
	
    print(f"Model\n {encoder}")
    print(f"{decoder}")
	
	# Loss and optimizer

    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum)
	
    total_step = len(train_loader)
	
    tota_loss_val = []
    tota_loss_train = []
    totat_epoch = []

    for epoch in range(0,total_epoch):
		
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass

        loss_tr = 0.0
		
		# switchin to training mode
        encoder.train()
        decoder.train()
		    
        for i, (images, captions, lengths) in enumerate(train_loader):

            decoder.zero_grad()
            encoder.zero_grad()
		    # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

		   # print("cp: {} target {}".format(captions.shape, targets.shape))

		    # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
		    
		   # print(" before loss cp: {} target {}".format(outputs.shape, targets.shape))
		   # Calculating loss and backprop
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            loss_tr = loss.item()

	#         t.set_postfix(loss=loss.item() )
	#         t.update()

		        # Print log info
            if i % log_steps == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
		              .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 


            tota_loss_train.append(loss_tr)
            totat_epoch.append(str(epoch))

	#         # Save the model checkpoints
	#         if (i+1) % save_step == 0:
	#             torch.save(decoder.state_dict(), os.path.join(
	#                 model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
	#             torch.save(encoder.state_dict(), os.path.join(
	#                 model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
		
		# Save the model checkpoints after every 50 epoch
        if (epoch+1) % 50 ==0:
            torch.save(decoder.state_dict(), os.path.join(model_path, 'decoder-{}.ckpt'.format(epoch+1)))
            torch.save(encoder.state_dict(), os.path.join(model_path, 'encoder-{}.ckpt'.format(epoch+1)))



		#################################################################
		#                    Validtion                                  #
		#################################################################


        # switchinit to evaluations mode
        encoder.eval()
        decoder.eval()
        loss_val = []
        for j, (images, captions, lengths) in enumerate(valid_loader):



            with torch.no_grad():
		        

		        # Set mini-batch dataset
                images = images.to(device)
                captions = captions.to(device)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

		        # Forward, backward and optimize
                features = en_coder(images)
                outputs = de_coder(features, captions, lengths)
                losses = criterion(outputs, targets)

		        # adding loss for each batch
                loss_val.append(losses)
		    
                avg_loss = torch.mean(torch.Tensor(loss_val))
		    
        tota_loss_val.append(avg_loss)
        print('\t\t Validation loss: %d %f' %(epoch+1, avg_loss))
        print("\n")

	





if __name__ == '__main__':


	# make a folder where model will be saved
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)


    train(vocab)


