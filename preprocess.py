from ont_fast5_api.fast5_interface import get_fast5_file
import os
import numpy as np
import glob

my_file_pos = open("gt_pos.txt", "r")
posli = my_file_pos.readlines()
my_file_pos.close()
posli = [pi.split('\n')[0] for pi in posli]

my_file_neg = open("gt_neg.txt", "r")
negli = my_file_neg.readlines()
my_file_neg.close()
negli = [pi.split('\n')[0] for pi in negli]


print("##### posli and negli length")
print(len(posli))
print(len(negli))
print()

arr = []
arrpos = []
name_pos = []
name_neg = []
i = 0
pi = 0

for fileNM in glob.glob('fast5/*.fast5'):
	with get_fast5_file(fileNM, mode="r") as f5:
		print("##### file: " + fileNM)
		for read in f5.get_reads():
			raw_data = read.get_raw_data(scale=True)
			if len(raw_data) >= 4500:
				if read.read_id in posli:
					pi += 1
					name_pos.append(read.read_id)
					arrpos.append(raw_data[1500:4500])
					if (pi%1000 == 0) and (pi != 0):
						print("##### 1000 pi: " + str(pi))
						print(np.array(arrpos).shape)
						print()
						np.save('npy/pos_' + str(pi), np.array(arrpos))
						with open('npy/name_pos_' + str(pi) + '.txt', 'w') as f:
							for item in name_pos:
								f.write("%s\n" % item)

						del arrpos
						del name_pos
						arrpos = []
						name_pos = []

				if read.read_id in negli:
					i += 1
					name_neg.append(read.read_id)
					arr.append(raw_data[1500:4500])
					if (i%1000 == 0) and (i != 0):
						print("##### 1000 i: " + str(i))
						print(np.array(arr).shape)
						print()
						np.save('npy/neg_' + str(i), np.array(arr))
						with open('npy/name_neg_' + str(i) + '.txt', 'w') as f:
							for item in name_neg:
								f.write("%s\n" % item)

						del arr
						del name_neg
						arr = []
						name_neg = []
