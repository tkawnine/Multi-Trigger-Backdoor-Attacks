import os
import subprocess
from torchvision.datasets import CIFAR10
from PIL import Image

# 1. Download CIFAR-10
dataset = CIFAR10(root='./data', train=True, download=True)

# 2. Create output directory
output_dir = './cifar10_stego'
os.makedirs(output_dir, exist_ok=True)

# 3. Create a sample secret file to hide
secret_file = './secret.txt'
with open(secret_file, 'w') as f:
    f.write('Confidential CIFAR data payload')

passphrase = 'original picture'

# 4. Iterate through images and apply steghide
for idx, (img, label) in enumerate(dataset):
    # Save image as JPEG (Steghide compatible)
    img_path = os.path.join(output_dir, f'img_{idx}_class_{label}.jpg')
    img.save(img_path, 'JPEG')
    
    # Run steghide command via subprocess
    cmd = [
        'steghide', 'embed', 
        '-cf', img_path, 
        '-ef', secret_file, 
        '-p', passphrase, 
        '-f'  # Force overwrite/embedding
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'Failed on image {idx}: {result.stderr}')
    else:
        print(f'Successfully embedded data in {img_path}')
