import time
import torch
import torchvision

import os.path
import numpy as np

import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

from vae import VAE
##################################

batch_size = 256
batch_size_test = 1000 


###################################
#Image loading and preprocessing
###################################

trainLoader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST('./data', train=True, download=True,
		transform=torchvision.transforms.ToTensor()),
		# Usually would do a normalize, but for some reason this messes up the output
	batch_size=batch_size, shuffle=True)

testLoader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST('./data', train=False, download=True,
		transform=torchvision.transforms.ToTensor()),
	batch_size=batch_size_test, shuffle=True)


num_epochs = 5

model = VAE().cuda()

# We are using a Sigmoid layer at the end so we must use CE loss. Why?
# ---> Rather, paper said to use CE loss.
def lossFun(x, x_prime, mu, logvar):
	binary_cross_entropy = F.binary_cross_entropy(x, x_prime, reduction='sum')
	
	distance_from_standard_normal = 0
	return binary_cross_entropy + distance_from_standard_normal

optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-3)




for epoch in range(num_epochs):
	# TrainLoader is a generator
	start = time.time()
	for data in trainLoader:		
		x, _ = data # Each 'data' is an image, label pair
		x = Variable(x).cuda() # Input image must be a tensor and moved to the GPU			

		# Forward pass
		x_prime, mu, logvar = model(x) # pass through into a reconstructed image
		x = x.view(-1, 28*28)		
		loss = lossFun(x_prime, x, mu, logvar)

		# Backward pass
		optimizer.zero_grad() # Backward function accumulates gradients, so we don't want to mix up gradients. 
				      # Set to zero instead.
		loss.backward()
		optimizer.step()
	elapsed = time.time() - start
	print('epoch [{}/{}], loss:{:.4f}, time:{:.2f}'.format(epoch+1, num_epochs, loss.data, elapsed))

torch.save(model, './checkpoints/model.pt')


#######################
# Testing
#######################


images, labels = iter(testLoader).next()
#print(labels)
images = Variable(images).cuda()
reconstructions, mu, logvar = model(images)
reconstructions = reconstructions.view(-1, 1, 28, 28)


# Display images / reconstructions
from matplotlib import pyplot as plt
def show(image):
	plt.imshow(image.permute(1, 2, 0))
	plt.show()

def show10(images1, images2):
	f, axes = plt.subplots(10, 2)
	for i in range(10):
		axes[i,0].imshow(images1.numpy()[i][0], cmap='gray')
		axes[i,1].imshow(images2.numpy()[i][0], cmap='gray')
	plt.show()

x  = images
x_ = reconstructions

show10(x.cpu(), x_.cpu().detach())







		
	














    



















