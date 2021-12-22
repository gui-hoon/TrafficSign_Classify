# version 1.3 : remove outliers at the line(out_j)
# inference time 60msec
# transpose time 3~8msec - why not constant?? this uses CPU
# image receive time 3msec
import torch, os, cv2
from torch._C import ParameterDict
from torchvision.transforms.transforms import ToPILImage
import torchvision.transforms as transforms
from multiprocessing import Process, Queue, Pipe, Value
import multiprocessing as mp
# import torchvision
# from PIL import Image
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch.nn as nn
import scipy.special, tqdm
import numpy as np
# import torchvision.transforms as transforms
# from opencv_transforms import transforms as transform
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
import time
from PIL import Image
from torch2trt import torch2trt, TRTModule

def detect_lane(conn, flag):
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start loading model...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
        row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError

    net_trt = TRTModule()
    net_trt.load_state_dict(torch.load(cfg.test_model))

    net_trt.eval()
    col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    print("start lane detection....")
    while(True):
        st = time.perf_counter()
        while(True) :
            if flag.value == 0 :
                break        
        imgr = conn.recv()

        st1 = time.perf_counter()
        # imgr = np.transpose(imgr, (2,0,1))
        # imgr = imgr.transpose((2,0,1))/255
        # img = transforms.ToTensor(imgr)/255

        img = torch.from_numpy(imgr).type(torch.FloatTensor).permute(2,0,1)/255 # HalfTensor) #
        img = img.unsqueeze_(0)
        st2 = time.perf_counter()
                
        img = img.cuda()
        with torch.no_grad():
            out = net_trt(img)      #120msec -> 100msec(F32) -> 60msec(F16, INT8)
        
        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]

        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg.griding_num] = 0
        out_j = loc
        end = time.perf_counter()
        conn.send(col_sample_w)
        conn.send(out_j)
        flag.value = 1
        end1 = time.perf_counter()
        print("recv:", st1-st, "tpose: ", st2-st1, "Inference time: ", end-st2, "send:", end1-end)

def main():
    args, cfg = merge_config()
    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
        row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError
    flag = Value('i', 0)    # Sync flag between main and detect_lane

    parent_conn, child_conn = Pipe()
    p = Process(target=detect_lane, args=(child_conn, flag))
    # p.daemon = True
    p.start()

    # img_w, img_h = 1280, 720 #1640, 590
    img_w, img_h = 800, 450 #1640, 590
    #row_anchor = culane_row_anchor
    num = 0
    tot = 0

    capfile = 'filesrc location=/home/adas/Documents/'+cfg.test_data+' ! qtdemux ! queue ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx,width=800,height=450 ! queue ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink'
    capfile1 = 'filesrc location=/home/adas/Documents/'+cfg.test_data+' ! qtdemux ! queue ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx,width=800,height=288 ! queue ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink'

    cap = cv2.VideoCapture(capfile, cv2.CAP_GSTREAMER)
    cap1 = cv2.VideoCapture(capfile1, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("There is no video stream")
        exit()
    
    res, imgs = cap.read()
    res1, img = cap1.read()
    # img = torch.from_numpy(img).type(torch.FloatTensor).permute(2,0,1)/255 # HalfTensor) #
    out_j_old = 0  # for removal of outliers of the lane detection points
    parent_conn.send(img)
    clane = np.int(cfg.num_lanes/2)
    
    while(cap.isOpened()):

        st = time.perf_counter()
        res, imgs = cap.read()
        res1, img = cap1.read()

        if not res:
            break

        if num == 0 :
            col_sample_w = parent_conn.recv()
            out_j = parent_conn.recv()
            # img = torch.from_numpy(img).type(torch.FloatTensor).permute(2,0,1)/255 # HalfTensor) #
            print("out_j", out_j.shape, out_j[0:3,2:8] * col_sample_w * img_w / 800)
            while(True):
                if flag.value == 1:
                    flag.value = 0
                    parent_conn.send(img)
                    break

        # center_w = abs((out_j[1,2]+out_j[1,1])* col_sample_w * img_w / 1600 - 640)
        center_w = np.mean((out_j[1:6,clane]+out_j[1:6,clane-1])* col_sample_w / 2 - 400) # 400 is half of 800(width) or Center of Camera
        if (center_w > 50):
            print("Warning Too Close Left Lane")
        elif (center_w < -50):
            print("Warning Too Close Right Lane")
        else :
            print("Good Lane Following")
            
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        # ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                        ppp = (int(out_j[k, i] * col_sample_w) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                        cv2.circle(imgs,ppp,5,(0,255,0),-1)

        end1 = time.perf_counter()
        # time.sleep(0.025)
                   
        cv2.imshow("Tracking", imgs)
        if cv2.waitKey(1) == 27:
           break

        end2 = time.perf_counter()
        print("time:", num, end2-st, tot)  

        num = num + 1
        num = num%2
        tot = tot + 1

    p.terminate()
    p.join()
    cap.release()
    cap1.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
