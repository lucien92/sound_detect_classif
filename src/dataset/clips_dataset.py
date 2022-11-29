import glob

from ..config import config

def clips_dataset():
	# List all .txt files in the dataset (annotations)
	paths = glob.glob(f"{config['dataset_path']}/*.txt")
	
	dataset = []
	for path in paths:
		dataset += get_labels(path)
	
	return dataset
		

def get_labels(path):
	with open(path) as f:
		lines 		= f.read().splitlines() 
		n_labels 	= len(lines) // 2
		labels 		=  []

		for i in range(n_labels):
			(start_t, end_t, name) 	= lines[i*2].split('\t')
			# print(name, '\n')
			(_, start_f, end_f) 	= lines[i*2 + 1].split('\t')
			labels.append({
				'path': path[:-3] + 'wav',
				'name': name[:-4] if name[-1].isdigit() else name,
				# 'name': name[:-4],
				'start_t': float(start_t),
				'end_t': float(end_t),
				'start_f': float(start_f),
				'end_f': float(end_f) 
			})

		return labels