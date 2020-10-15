import torch
from torch.nn import functional as F
import cv2
import numpy as np

class GradCam:
  """ 
  GradCam Implementation
  Args:
    model: NN architecture
    layer: Track gradients and activation outputs from
           this layer of the model
    device (str): cpu/gpu
  """
  def __init__(self, model, layer, device):
    self.model = model.to(device)
    self.grads = {}
    self.activations = {}
    self.device = device

    layer.register_forward_hook(self.forward_hook)
    layer.register_backward_hook(self.backward_hook)

  def forward_hook(self, mod, node_inp, node_out):
      self.activations['value'] = node_out
    
  def backward_hook(self, mod, grad_inp, grad_out):
    self.grads['value'] = grad_out

  def __call__(self, input, cls=None):
    """
    Args:
      input: input tensor image
      class (optional): index of class on which gradients are to be calculated
                        if none then the grads are calculated based on the best score class
    Return:
      map: weighted sum of activation maps from the 'layer' for input image
      output: model output on image
    """
    inp_sz = input.shape  # input tensor shape
    self.model.eval()     # to access every node of the model

    output = self.model(input.to(self.device))
    if not cls:
      cls = torch.argmax(output, dim=1).item()

    class_score = output[:, cls]    # model score for given class
    self.model.zero_grad()      # cleaning any accumulated gradients
    class_score.backward()  # claculating gradient with respect to class_score
    
    # global average pooling
    self.grads['value'] = self.grads['value'][0]
    grad_sz = self.grads['value'].size()
    alpha = self.grads['value'].view(grad_sz[0], grad_sz[1], -1)  \
            .mean(2).unsqueeze(-1).unsqueeze(-1)                       # increasing dims to match dims of activations
    
    # applying relu to weighted sum of activations
    map = F.relu((alpha*self.activations['value']).sum(dim = 1, keepdim = True))
    # resizing activations to the size of input image
    map = F.interpolate(map, size = (inp_sz[-2], inp_sz[-2]), mode = 'bilinear', align_corners=False)
    map = (map-map.min())/map.max()   # scaling for larger variance
    return map, output
  
  
def getHeatMap(map, image, STD=(0.229, 0.224, 0.225), MEAN=(0.485, 0.456, 0.406)):
  """
  Function to create heatmap images from mask generated using Gradcam
  Args:
    image: tensor images,
           expected dim = (1 x channels x H x W)
    map (tensor): output of Gradcam
  Return:
    heatmap: visual result of Gradcam
  """
  heatmap = map.detach().to('cpu').numpy().squeeze(0).transpose(1, 2, 0).astype(np.uint8)  # denormalizing tensor and converting to numpy array
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)/255.
  r, g, b = [heatmap[:, :, i] for i in range(0, 3)]
  heatmap = np.stack((b, g, r), axis = -1)

  image = image.detach().to('cpu').numpy().squeeze(0).transpose(1, 2, 0)
  image = (image*np.array(STD) + np.array(MEAN)).clip(0, 1)
  # overlaying heatmap on image
  heatmap = (heatmap + image)/(heatmap + image).max() 
  return heatmap