# !/usr/bin/env python3
# coding=utf-8

import torch
import os
from pose_estimation import *

def evaluate(model, val_data_dir='./data'):
    box_size = 368
    scale_search = [0.5, 1.0, 1.5, 2.0]
    param_stride = 8

    # Predict pictures
    list_dir = os.walk(val_data_dir)
    for root, dirs, files in list_dir:
        for f in files:
            test_image = os.path.join(root, f)
            print("test image path", test_image)
            img_ori = cv2.imread(test_image)  # B,G,R order

            multiplier = [scale * box_size / img_ori.shape[0] for scale in scale_search]

            for i, scale in enumerate(multiplier):
                h = int(img_ori.shape[0] * scale)
                w = int(img_ori.shape[1] * scale)
                pad_h = 0 if (h % param_stride == 0) else param_stride - (h % param_stride)
                pad_w = 0 if (w % param_stride == 0) else param_stride - (w % param_stride)
                new_h = h + pad_h
                new_w = w + pad_w

                img_test = cv2.resize(img_ori, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                img_test_pad, pad = pad_right_down_corner(img_test, param_stride, param_stride)
                img_test_pad = np.transpose(np.float32(img_test_pad[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5

                feed = Variable(torch.from_numpy(img_test_pad))
                output1, output2 = model(feed)
                print(output1.shape, output2.shape)


# loading model
state_dict = torch.load('./models/coco_pose_iter_440000.pth.tar')['state_dict']

# create a model instance
model_fp32 = get_pose_model()
model_fp32.load_state_dict(state_dict)
model_fp32.float()

# model must be set to eval mode for static quantization logic to work
model_fp32.eval()

# attach a global qconfig, which contains information about what kind
# of observers to attach. Use 'fbgemm' for server inference and
# 'qnnpack' for mobile inference. Other quantization configurations such
# as selecting symmetric or assymetric quantization and MinMax or L2Norm
# calibration techniques can be specified here.
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Prepare the model for static quantization. This inserts observers in
# the model that will observe activation tensors during calibration.
model_fp32_prepared = torch.quantization.prepare(model_fp32)

# calibrate the prepared model to determine quantization parameters for activations
# in a real world setting, the calibration would be done with a representative dataset
evaluate(model_fp32_prepared)

# Convert the observed model to a quantized model. This does several things:
# quantizes the weights, computes and stores the scale and bias value to be
# used with each activation tensor, and replaces key operators with quantized
# implementations.
model_int8 = torch.quantization.convert(model_fp32_prepared)
print("model int8", model_int8)
# save model
torch.save(model_int8.state_dict(),"./openpose_vgg_quant.pth")
