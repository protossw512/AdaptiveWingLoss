import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from core import models
import matplotlib.pyplot as plt


def get_preds_fromhm(hm, center=None, scale=None, rot=None):
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-0.5)

    return preds


# Parse arguments
parser = argparse.ArgumentParser()

# Checkpoint and pretrained weights

parser.add_argument('--pretrained_weights', type=str,
                    help='a directory to save pretrained_weights')

# Network parameters
parser.add_argument('--num_landmarks', type=int, default=98,
                    help='Number of landmarks')
parser.add_argument('--hg_blocks', type=int, default=4,
                    help='Number of HG blocks to stack')
parser.add_argument('--gray_scale', type=str, default="False",
                    help='Whether to convert RGB image into gray scale during training')
parser.add_argument('--end_relu', type=str, default="False",
                    help='Whether to add relu at the end of each HG module')
parser.add_argument('--image_source_loc', type=str, default="True",
                    help='image source location')
parser.add_argument('--image_target_loc', type=str, default="True",
                    help='image target location')

OUTPUT_LOC
args = parser.parse_args()

PRETRAINED_WEIGHTS = args.pretrained_weights
IMAGE_SOURCE_LOCATION = args.image_source_loc
IMAGE_TARGET_LOCATION = args.image_target_loc
GRAY_SCALE = False if args.gray_scale == 'False' else True
HG_BLOCKS = args.hg_blocks
END_RELU = False if args.end_relu == 'False' else True
NUM_LANDMARKS = args.num_landmarks

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

use_gpu = torch.cuda.is_available()
model_ft = models.FAN(HG_BLOCKS, END_RELU, GRAY_SCALE, NUM_LANDMARKS)

if PRETRAINED_WEIGHTS != "None":
    checkpoint = torch.load(PRETRAINED_WEIGHTS)
    if 'state_dict' not in checkpoint:
        model_ft.load_state_dict(checkpoint)
    else:
        pretrained_weights = checkpoint['state_dict']
        model_weights = model_ft.state_dict()
        pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                              if k in model_weights}
        model_weights.update(pretrained_weights)
        model_ft.load_state_dict(model_weights)

model = model_ft.to(device)

model.eval()
# Iterate over data.
resolution = 256
image = np.array(Image.open(IMAGE_LOCATION))
new_image = cv2.resize(image, dsize=(int(resolution), int(resolution)), interpolation=cv2.INTER_LINEAR)
images_list = [new_image]

with torch.no_grad():
    for data_number, data in enumerate(images_list):

        data = data.transpose((2, 0, 1)) #/255.0
        data = data[np.newaxis, ...]

        # get the inputs
        inputs = torch.from_numpy(data).float().div(255.0) #dtype=torch.FloatTensor)
        # inputs = inputs.type(torch.cuda.DoubleTensor)
        # wrap them in Variable
        if use_gpu:
            inputs = inputs.to(device)
        else:
            inputs = Variable(inputs)

        outputs, boundary_channels = model(inputs)
        for i in range(inputs.shape[0]):
            img = inputs[i]
            img = img.cpu().numpy()
            img = img.transpose((1, 2, 0)) #*255.0   CHECK IF NECESSERY!
            img = img.astype(np.uint8)
            img = Image.fromarray(img)

            # pred_heatmap = outputs[-1][i].detach().cpu()[:-1, :, :]
            pred_heatmap = outputs[-1][:, :-1, :, :][i].detach().cpu()
            pred_landmarks = get_preds_fromhm(pred_heatmap.unsqueeze(0))
            pred_landmarks = pred_landmarks.squeeze().numpy()

            fig = plt.figure()
            im = plt.imread(IMAGE_SOURCE_LOCATION)
            image_resized_256 = im.resize((227, 227))

            impolt = plt.imshow(im)
            for x, y in pred_landmarks:
                plt.scatter(x*float(im.shape[0])/64, y*float(im.shape[1])/64, c='r')


            plt.savefig(IMAGE_TARGET_LOCATION, bbox_inches='tight',pad_inches=0)
