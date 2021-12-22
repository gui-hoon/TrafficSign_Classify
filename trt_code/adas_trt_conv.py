import torch, os, cv2
from torchvision.transforms.transforms import ToPILImage
# import torchvision
# from PIL import Image
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch.nn as nn
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
#from opencv_transforms import transforms as transform
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
import time
from PIL import Image
from torch2trt import torch2trt, TRTModule

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
        row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError

    net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane,cfg.num_lanes),
                    use_aux=False).cuda() # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_w, img_h = 1280, 720 #1640, 590
    # tmp = np.zeros((3,288,800))
    num = 0

    cap = cv2.VideoCapture('filesrc location=/home/adas/Documents/test_data/highway_D1_Trim.mp4 ! qtdemux ! queue ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw,format=BGRx,width=800,height=288 ! queue ! videoconvert ! queue ! video/x-raw, format=BGR ! appsink', cv2.CAP_GSTREAMER)

    if(cap.isOpened()):

        st = time.perf_counter()
        res, img = cap.read()
        
        img = img/255
        img = np.transpose(img, (2,0,1))

        img = torch.from_numpy(img).type(torch.FloatTensor)
        img = img.unsqueeze_(0)
            
        img = img.cuda()
        #net_trt = torch2trt(net, [img])
        net_trt = torch2trt(net, [img], fp16_mode=True)
        #net_trt = torch2trt(net, [img], int8_mode=True)
        with torch.no_grad():
            out = net(img)      #120msec
            out_trt = net_trt(img)

        print(torch.max(torch.abs(out - out_trt)))

#torch.save(net_trt.state_dict(), 'culane_18_10trt.pth')
torch.save(net_trt.state_dict(), 'culane_18_10trt_fp16.pth')
#torch.save(net_trt.state_dict(), 'culane_18_10trt_int8.pth')

# model_trt = TRTModule()

# model_trt.load_state_dict(torch.load('alexnet_trt.pth'))


