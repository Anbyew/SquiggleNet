import time
start_time = time.time()
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.nn import functional as F
from scipy import stats
from ont_fast5_api.fast5_interface import get_fast5_file

import numpy as np
import os


#device = torch.device('cpu')
#device = torch.device('cuda:0')
if torch.cuda.is_available:device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device: " + str(device))

########################
#### Load the model ####
########################

def conv3(in_channel, out_channel, stride=1, padding=1, groups=1):
  return nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, 
				   padding=padding, bias=False, dilation=padding, groups=groups)

def conv1(in_channel, out_channel, stride=1, padding=0):
  return nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride, 
				   padding=padding, bias=False)

def bcnorm(channel):
  return nn.BatchNorm1d(channel)


class Bottleneck(nn.Module):
	expansion = 1.5
	def __init__(self, in_channel, out_channel, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = conv1(in_channel, in_channel)
		self.bn1 = bcnorm(in_channel)
		self.conv2 = conv3(in_channel, in_channel, stride)
		self.bn2 = bcnorm(in_channel)
		self.conv3 = conv1(in_channel, out_channel)
		self.bn3 = bcnorm(out_channel)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride
  
	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class ResNet(nn.Module):
	def __init__(self, block, layers, num_classes=2):
		super(ResNet, self).__init__()
		self.chan1 = 20

		# first block
		self.conv1 = nn.Conv1d(1, 20, 19, padding=5, stride=3)
		self.bn1 = bcnorm(self.chan1)
		self.relu = nn.ReLU(inplace=True)
		self.pool = nn.MaxPool1d(2, padding=1, stride=2)
		self.avgpool = nn.AdaptiveAvgPool1d(1)
		self.fc = nn.Linear(67, 2)

		self.layer1 = self._make_layer(block, 20, layers[0])
		self.layer2 = self._make_layer(block, 30, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 45, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 67, layers[3], stride=2)
		#self.layer5 = self._make_layer(block, 100, layers[4], stride=2)

		# initialization
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm1d)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	
	def _make_layer(self, block, channels, blocks, stride=1):
		downsample = None
		if stride != 1 or self.chan1 != channels:
			downsample = nn.Sequential(
				conv1(self.chan1, channels, stride),
				bcnorm(channels),
			)

		layers = []
		layers.append(block(self.chan1, channels, stride, downsample))
		if stride != 1 or self.chan1 != channels:
		  self.chan1 = channels
		for _ in range(1, blocks):
			layers.append(block(self.chan1, channels))

		return nn.Sequential(*layers)

	def _forward_impl(self, x):
		x = x.unsqueeze(1)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.pool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		#x = self.layer5(x)

		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)

		return x

	def forward(self, x):
	  return self._forward_impl(x)


tpname = 'model_B4_2_3000_totmad_br_2.ckpt'
bmodel = ResNet(Bottleneck, [2,2,2,2]).to(device).eval()
tpp = torch.load(os.path.join('', tpname),  map_location=device)
bmodel.load_state_dict(tpp)

print("[Step 0]$$$$$$$$$$ Done loading model......")


##### Normalization #####
def normalization(data_test, batchi):
	mad = stats.median_abs_deviation(data_test, axis=1, scale='normal')
	m = np.median(data_test, axis=1)   
	data_test = ((data_test - np.expand_dims(m,axis=1))*1.0) / (1.4826 * np.expand_dims(mad,axis=1))

	x = np.where(np.abs(data_test) > 3.5)
	for i in range(x[0].shape[0]):
		if x[1][i] == 0:
			data_test[x[0][i],x[1][i]] = data_test[x[0][i],x[1][i]+1]
		elif x[1][i] == 2999:
			data_test[x[0][i],x[1][i]] = data_test[x[0][i],x[1][i]-1]
		else:
			data_test[x[0][i],x[1][i]] = (data_test[x[0][i],x[1][i]-1] + data_test[x[0][i],x[1][i]+1])/2

	data_test = torch.tensor(data_test).float()

	print("[Step 2]$$$$$$$$$$ Done data normalization with batch "+ str(batchi))
	return data_test

########################
####### Run Test #######
########################
def process(data_test, batchi):
	#if torch.cuda.is_available:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	#device = torch.device('cpu')
	#print("Device again: " + str(device))

	with torch.no_grad():
		testx = data_test.to(device)
		outputs_test = bmodel(testx)
		np.savetxt('me_output/ba_' + str(batchi) + '.txt', outputs_test.max(dim=1).indices.int().data.cpu().numpy())
		print("[Step 3]$$$$$$$$$$ DONE processing with batch "+ str(batchi))
		print(outputs_test.shape)
		del outputs_test

########################
#### Load the data #####
########################

batchsize = 1
f5path = '../fast5'#../fast5
#f5path = '/media/yuweibao/71fddc75-d30f-404c-b9ad-0def01cd1368/genomes/fast5/nazm50/34/fast5'

j = 0
def get_raw_data(fileNM, data_test):
	fast5_filepath = os.path.join(f5path, fileNM)
	with get_fast5_file(fast5_filepath, mode="r") as f5:
		for read in f5.get_reads():
			raw_data = read.get_raw_data(scale=True)
			if len(raw_data) > 5000:
				data_test.append(raw_data[2000:5000])
				j += 1
				if j >= 500:
					return data_test
	return data_test

directory = os.fsencode(f5path)
data_test = []
batchi = 0
it = 0
mid_time1 = time.time()
for file in os.listdir(directory):
	filename = os.fsdecode(file)
	if filename.endswith(".fast5"): 
		data_test = get_raw_data(filename, data_test)
		it += 1

		if it == batchsize:
			print("[Step 1]$$$$$$$$$$ Done loading data with batch " + str(batchi)+ ", Getting " + str(len(data_test)) + " of sequences...")
			mid_time2 = time.time()
			data_test = normalization(data_test, batchi)
			process(data_test, batchi)
			print("[Step 4]$$$$$$$$$$ Done with batch " + str(batchi) + ", total time consumed: " + str(time.time() - mid_time1) + \
				", loading time: " + str(mid_time2 - mid_time1) + ", processing time: " + str(time.time() - mid_time2))
			print()
			del data_test
			data_test = []
			batchi += 1
			it = 0
			mid_time1 = time.time()

print("[Step FINAL]--- %s seconds ---" % (time.time() - start_time))
