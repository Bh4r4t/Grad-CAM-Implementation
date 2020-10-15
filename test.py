import torch
from torchvision import transforms as tfms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from gradcam import GradCam, getHeatMap

imgpath = "./german_shepherd.jpg"  
device = 'cpu'

STD=(0.229, 0.224, 0.225)
MEAN=(0.485, 0.456, 0.406)

img = Image.open(imgpath)
img_t = tfms.Compose([tfms.Resize((375, 375)), tfms.ToTensor(), tfms.Normalize(MEAN, STD)])
img = img_t(img)

# loading saved model
path = "./saved_model.pth"
saved = torch.load(path, device)
net = saved['model']

print("Model Loaded Successfully!")

# track gradients from given layer
layer = net.layer4
gc = GradCam(net, layer, device)

gc_out = gc(img.unsqueeze(0))
hm = getHeatMap(gc_out[0]*255., img.unsqueeze(0))

print("GradCam Computation done!")

# denorm image
image = img.detach().to('cpu').numpy().transpose(1, 2, 0)
image = (image*np.array(STD) + np.array(MEAN)).clip(0, 1)

# display result
fig, ax = plt.subplots(1, 2, figsize = (8,8))
for x in ax:
  x.set_xticks([])
  x.set_yticks([])
ax[0].imshow(hm)
ax[1].imshow(image)
fig.show()