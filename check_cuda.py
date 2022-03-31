import torch

if torch.cuda.is_available():
	print('cuda is available')
	for i in range(torch.cuda.device_count()):
		device = torch.cuda.device(i)
		print(i, torch.cuda.get_device_name(device))
else:
	print('cuda unavailable')