from torch2trt import torch2trt
from models import *
from utils.datasets import *
from utils.utils import *
import argparse
#import torch

#data = torch.randn((1, 3, 224, 224)).cuda().half()
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='truck.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='truck.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    args = parser.parse_args()
    return args

args = get_parser()
img_size = args.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
source = args.source
weights = args.weights

device = torch_utils.select_device(args.device)
model = Darknet(args.cfg, img_size)
model.load_state_dict(torch.load(weights, map_location=device)['model'])
model.to(device).eval().half()
dataset = LoadImages(source, img_size = img_size, half = True)

for path, img, im0s, vid_cap in dataset:
    t = time.time()
    img = torch.from_numpy(img).to(device) # numpy conversion to torch gpu tensor
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img)[0].float() # prediction outcome from passing in the input image
    pred = non_max_suppression(pred, args.conf_thres, args.nms_thres)

model_trt = torch2trt(model, img[0], fp16_mode=True)
output_trt = model_trt(img)
output = pred
print(output.flatten()[0:10])
print(output_trt.flatten()[0:10])
print('max error: %f' % float(torch.max(torch.abs(output - output_trt))))
torch.save(model_trt.state_dict(), 'truck_trt.pth')



#from torch2trt import TRTModule

#model_trt = TRTModule()

#model_trt.load_state_dict(torch.load('resnet18_trt.pth'))


#####################################################################
