import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

# cumulative counting
from centroidandcorrelationcounting import *

def detect(save_txt=False, save_img=False):

    # inserted parameters
    img_size = opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out = opt.output
    source = opt.source
    weights = opt.weights
    half = opt.half
    view_img = opt.view_img

    # converge sources for detection to webcam
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    video = source.endswith('.mp4')

    # Initialize computational device
    device = torch_utils.select_device(opt.device)
    #initialize output folder
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder


    # Initialize model from model.py import *
    model = Darknet(opt.cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # send model to gpu
    model.to(device).eval()

    # conver model to Half precision (16-bits)
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path = None
    vid_writer = None

    if webcam:
        view_img = True
    #    save_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)

    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(opt.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

###########

    #width, height = img_size
    centroid_tracker = centroidtracker(maxdisappear = 40, maxdistance = 300)
    trackers = []
    trackableobjects = {}

    totalframes = 0
    totalupdown = 0
    skip_frames = 30

    fps1 = FPS().start()

############

    # Run inference

    t0 = time.time() # initial time

    for path, img, im0s, vid_cap in dataset:
        t = time.time()

        rects = []
        img = torch.from_numpy(img).to(device) # numpy conversion to torch gpu tensor

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        (width, height) = img.shape[:2]

        if webcam:
            for i, v in enumerate(img):
                p, im0 = path[i], im0s[i]
#            p, s, im0 = path[i], '%g: ' % i, im0s[i]
        else:
            p, im0 = path, im0s
#            p, s, im0 = path, '', im0s
            save_path = str(Path(out) / Path(p).name)
#            s += '%gx%g ' % img.shape[2:]  # add to string image dimensions
        if totalframes % skip_frames == 0:
            trackers = []
            # Get detections

            pred = model(img)[0] # prediction outcome from passing in the input image

            if opt.half:
                pred = pred.float()

            # Apply NMS on prediction, this suppresses the bounding box to converge based on the IOU threshold
            pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
#                    p, s, im0 = path[i], '%g: ' % i, im0s[i] # path, string, inputframe
                    s = '%g: ' % i
                else:
#                    p, s, im0 = path, '', im0s
                    s = ''

#                    save_path = str(Path(out) / Path(p).name)
                    s += '%gx%g ' % img.shape[2:]  # add to string image dimensions

#                if video:
#                    vid_cap = cv2.VideoCapture(source)
#                    im0 = im0.read()
#                    im0 = im0[1]
#                    im0 = imutils.resize(im0, 300)
#                    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)


                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, _, cls in det:

                        if classes[int(cls)] != 'person':
                            continue

                        if save_txt:  # Write to file
                            with open(save_path + '.txt', 'a') as file:
                                file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (classes[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

    ### implementation of counting########
    ######################################
                        font = cv2.FONT_HERSHEY_SIMPLEX

                        boxes = xyxy # must be normalized
                        for i,v in enumerate(boxes):
                            boxes[i] = int(v)

                        (x, y, xmax, ymax) = boxes

                        ############## crop images ############
                        crop_img = im0[y:ymax, x:xmax]
                        #######################################

                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(x, y, xmax, ymax)
                        tracker.start_track(im0, rect)

    #                    index = det[:, -1] # must convert to ones

                        scores = det[:, -3] # must be zero 2D array
                        scores = scores[:].tolist()
    #                    classes1 = classes1[:].tolist()

    ##                    print(boxes)
    ##                    print(scores)
    ##                    print(det)

                        trackers.append(tracker)

        else:
            for tracker in trackers:

                tracker.update(im0)
                pos = tracker.get_position()

                x = int(pos.left())
                y = int(pos.top())
                xmax = int(pos.right())
                ymax = int(pos.bottom())

                rects.append((x, y, xmax, ymax))


        font = cv2.FONT_HERSHEY_SIMPLEX
    #    cv2.putText(im0, 'ROI Line', (545, 240), font, 0.6, (0, 0, 0xFF), 2, cv2.LINE_AA,)

        objects = centroid_tracker.update(rects)

    ##################################

        for (objectID, centroid) in objects.items():

            to = trackableobjects.get(objectID, None)
            if to is None:
                to = trackedobject(objectID, centroid)

            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    totalupdown += 1
                    to.counted = True
#                    if direction < 0 and centroid[1] < height // 2:
#                        totalupdown += 1
#                        to.counted = True
#
#                    elif direction > 0 and centroid[1] > height // 2:
#                        totalupdown += 1
#                        to.counted = True

            trackableobjects[objectID] = to

            text = "ID {}".format(objectID)
            cv2.putText(im0, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(im0, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        cv2.putText(im0, 'Detected Objects: ' + str(totalupdown), (10,35), font, 0.8, (0,0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX,)

        totalframes +=1
        fps1.update()

        print('Done. (%.3fs)' % (time.time() - t))

        # Stream results
        if view_img:
            cv2.imshow(p, im0)
#            cv2.imshow("frame", im0)

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'images':
                cv2.imwrite(save_path, im0)
#                cv2.imwrite(save_path, crop_img)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'opt.fourcc'), fps, (w, h))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MPEG'), 30, (w, h))
                vid_writer.write(im0)

        print('%sDone. (%.3fs)' % (s, time.time() - t))
    fps1.stop()

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

    print("elapsed time: {:.2f}".format(fps1.elapsed()))
    print("approx. FPS: {:.2F}".format(fps1.fps()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
