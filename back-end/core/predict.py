import cv2
import torch
import numpy as np
from torch.autograd import Variable
from seg.BiSeNet.newtools.utils import label_img_to_color

def predict(dataset, model, ext):
    global img_y
    x = dataset[0].replace('\\', '/')
    file_name = dataset[1]
    print(x)
    print(file_name)
    x = cv2.imread(x)
    img_y, image_info = model.detect(x)
    # cv2.imshow('img',img_y)
    # cv2.waitKey(0)
    try:
        cv2.imwrite('./tmp/draw/{}.{}'.format(file_name, ext), img_y)
    except Exception as e:
        print(f"详细错误信息: {e}")
        # 可能还想记录 stack trace 或其他信息
        raise Exception('保存图片时出错.Error saving thepicture.')
    return image_info

def segment(dataset, model, ext):
    seg_type = ["Ridge", "Tree", "Crops", "BG",
                "BG", "BG", "BG", "BG"]
    global img_y
    x = dataset[0].replace('\\', '/')
    file_name = dataset[1]
    print(x)
    print(file_name)
    x = cv2.imread(x)
    # 数据预处理
    x = cv2.resize(x, (512, 512),
                         interpolation=cv2.INTER_NEAREST)
    x = x/255.0
    x = x - np.array([0.485, 0.456, 0.406])
    x = x/np.array([0.229, 0.224, 0.225]) 
    x = np.transpose(x, (2, 0, 1)) 
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    x = Variable(x).cuda()
    print(x.shape)
    model.eval()
    outputs,*outputs_aux = model(x)
    outputs = outputs.data.cpu().numpy() # (shape: (batch_size, num_classes, img_h, img_w))
    pred_label_imgs = np.argmax(outputs, axis=1) # (shape: (batch_size, img_h, img_w))
    pred_label_imgs = pred_label_imgs.astype(np.uint8)

    for i in range(pred_label_imgs.shape[0]):
        pred_label_img = pred_label_imgs[i]
        unique, counts = np.unique(pred_label_img, return_counts=True)
        class_areas = dict(zip(unique, counts))
        pred_label_img_color  = label_img_to_color(pred_label_img)
    print(class_areas)
    img_y = pred_label_img_color
    image_info = {}
    for class_id, area in class_areas.items():
        key = str(seg_type[class_id])
        image_info[key] = ['{}'.format(
                        area), np.round(area/(512*512), 5)]
    print(image_info)
    try:
        cv2.imwrite('./tmp/draw/{}.{}'.format(file_name, ext), img_y)
    except Exception as e:
        print(f"详细错误信息: {e}")
        raise Exception('保存图片时出错.Error saving thepicture.')
    return image_info

