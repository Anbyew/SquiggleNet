import time
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.nn import functional as F
from scipy import stats
from ont_fast5_api.fast5_interface import get_fast5_file
from model import ResNet

import numpy as np
import os

start_time = time.time()
if torch.cuda.is_available:device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device: " + str(device))


### load model
tpname = 'model_B4_2_3000_totmad_br_2.ckpt'
bmodel = ResNet(Bottleneck, [2,2,2,2]).to(device).eval()
tpp = torch.load(os.path.join('', tpname),  map_location=device)
bmodel.load_state_dict(tpp)

print("[Step 0]$$$$$$$$$$ Done loading model......")


########################
##### Normalization ####
########################
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
f5path = '../fast5'

j = 0
def get_raw_data(fileNM, data_test):
	fast5_filepath = os.path.join(f5path, fileNM)
	with get_fast5_file(fast5_filepath, mode="r") as f5:
		for read in f5.get_reads():
			raw_data = read.get_raw_data(scale=True)
			if len(raw_data) > 5000:
				data_test.append(raw_data[2000:5000])
				j += 1
	return data_test

directory = os.fsencode(f5path)
data_test = []
batchi = 0
it = 0
for file in os.listdir(directory):
	filename = os.fsdecode(file)
	if filename.endswith(".fast5"): 
		data_test = get_raw_data(filename, data_test)
		it += 1

		if it == batchsize:
			print("[Step 1]$$$$$$$$$$ Done loading data with batch " + str(batchi)+ ", Getting " + str(len(data_test)) + " of sequences...")
			data_test = normalization(data_test, batchi)
			process(data_test, batchi)
			print("[Step 4]$$$$$$$$$$ Done with batch ")
			print()
			del data_test
			data_test = []
			batchi += 1
			it = 0

print("[Step FINAL]--- %s seconds ---" % (time.time() - start_time))

