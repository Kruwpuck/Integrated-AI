import torch

# Load file biner tersebut
model_data = torch.load('CONFERENCE/efficientnet_b3_ultrasound.pth', map_location='cpu')

# Lihat kunci-kunci di dalamnya (biasanya berisi state_dict)
print(model_data.keys())

# Kalau mau lihat bobot tiap layer (isinya bakal penuh angka)
# print(model_data)