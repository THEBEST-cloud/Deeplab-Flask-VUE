import torch

from core import process, predict

from seg.BiSeNet.configs import cfg_factory
from seg.BiSeNet.lib.models import model_factory

def c_main(path, model, ext):
    image_data = process.pre_process(path)
    image_info = predict.predict(image_data, model, ext)

    return image_data[1] + '.' + ext, image_info

def seg_main(path, model, ext):
    image_data = process.pre_process(path)
    image_info = predict.segment(image_data, model, ext)
    return image_data[1] + '.' + ext, image_info

def get_model():
    cfg = cfg_factory["bisenetv2"]
    network = model_factory[cfg.model_type](8)
    network.cuda()
    network.load_state_dict(torch.load(r"seg\BiSeNet\training_logs\model\model_bisnet_higha3_epoch_196.pth"))
    return network

if __name__ == '__main__':
    pass
