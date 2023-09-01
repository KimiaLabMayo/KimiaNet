from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pdb
from torch.utils.data import Dataset, DataLoader
from glob import glob
from skimage import io, transform
import torch.nn.functional as F
from PIL import Image
import pickle	

plt.ion()   # interactive mode
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
save_address_1024 = './'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# making ready the test dataloader
test_folder_selection = glob('./Test_Patches/*')

trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
	'train': transforms.Compose([
        # transforms.Resize(1000),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': transforms.Compose([
        # transforms.Resize(1000),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
}


class My_dataloader(Dataset):
 
	def __init__(self, data_24, transform):
		"""
		Args:
			data_24: path to input data
		"""
		self.data_24 = data_24
		self.pathes_24 = glob(self.data_24+'/*')
		self.transform = transform
 
	def __len__(self):
		return len(self.pathes_24)
 
	def __getitem__(self, idx):
		img_24 = Image.open(self.pathes_24[idx]).convert('RGB')
		img_24_name = self.pathes_24[idx].split('/')[-1]
		img_24_folder = self.pathes_24[idx].split('/')[-2]
		if self.transform:
			img_24 = self.transform(img_24)
		return img_24, img_24_name, img_24_folder



def test_model(model, criterion, num_epochs=25):
	since = time.time()

	model.eval()   # Set model to evaluate mode

	running_loss = 0.0
	running_corrects = 0
	for i in range(1):
		slide_patches_dict_1024 = {}
		test_path = os.path.join('./Test_Patches/', test_folder_selection[i])
		test_path_test = './test_folder'
		test_imagedataset = My_dataloader(test_path_test,trans)
		dataloader_test = torch.utils.data.DataLoader(test_imagedataset, batch_size=16,shuffle=False, num_workers=7)
		counter = 0
		# Iterate over data.
		for ii, (inputs, img_name, folder_name) in enumerate(dataloader_test):
			inputs = inputs.to(device)
			output1, outputs = model(inputs)
			output_1024 = output1.cpu().detach().numpy()
			# output_128 = output2.cpu().detach().numpy()
			for j in range(len(outputs)):
				slide_patches_dict_1024[img_name[j]] = output_1024[j]
		outfile_1024 = open(save_address_1024+folder_name[0]+'_DenseNet121Features_dict.pickle','wb')
		pickle.dump(slide_patches_dict_1024, outfile_1024)
		outfile_1024.close() 
		time_elapsed = time.time() - since
		print('Evaluation completed in {:.0f}m {:.0f}s'.format(
			time_elapsed // 60, time_elapsed % 60))
	return model


class fully_connected(nn.Module):
	"""docstring for BottleNeck"""
	def __init__(self, model, num_ftrs, num_classes):
		super(fully_connected, self).__init__()
		self.model = model
		self.fc_4 = nn.Linear(num_ftrs,num_classes)

	def forward(self, x):
		x = self.model(x)
		x = torch.flatten(x, 1)
		out_1 = x
		out_3 = self.fc_4(x)
		return  out_1, out_3


model = torchvision.models.densenet121(pretrained=True)
for param in model.parameters():
	param.requires_grad = False
model.features = nn.Sequential(model.features , nn.AdaptiveAvgPool2d(output_size= (1,1)))
num_ftrs = model.classifier.in_features
model_final = fully_connected(model.features, num_ftrs, 30)
model = model.to(device)
model_final = model_final.to(device)
model_final = nn.DataParallel(model_final)
params_to_update = []
criterion = nn.CrossEntropyLoss()

model_final.load_state_dict(torch.load('./weights/KimiaNetPyTorchWeights.pth'))
model = test_model(model_final, criterion, num_epochs=1)