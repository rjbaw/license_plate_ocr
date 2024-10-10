import onnx
import onnx_tensorrt.backend as backend
import onnx.utils

import argparse
from sys import platform

from utils.datasets import *
from utils.utils import *

def detect():
	img_size = opt.img_size
	source = opt.source
	weights = opt.weights
	view_img = opt.view_img

	webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
#	device = torch_utils.select_device(opt.device)
	device = 'CUDA:0'
#	device = 'CPU'
#	if os.path.exists(out):
#		continue
#	os.makedirs(out)

	model = onnx.load(weights)
	onnx.checker.check_model(model)
	polished_model = onnx.utils.polish_model(model)
	engine = backend.prepare(polished_model, device = device)
	half = half and device.type != 'cpu'

	vid_path, vid_writer = None, None
	if webcam:
		view_img = True
		torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
		dataset = LoadStreams(source, img_size=img_size, half=half)
	else:
		save_img = True
		dataset = LoadImages(source, img_size=img_size, half=half)

	t0 = time.time()
	for path, img, im0s, vid_cap in dataset:
		t = time.time()

		input_data = img
		output_data = engine.run(input_data)[0]
		print(output_data)
		print(output_data.shape)

		print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()

    print(opt)
    detect()
