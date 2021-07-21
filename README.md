# Pytorch Model Quantization
Pose Estimation uses Pytorch for static quantization, saving, and loading of models

# Get data and model
Representative Dataset: You can get it from MSCOCO  [val2017.zip](http://images.cocodataset.org/zips/val2017.zip).

Model: You can get the model from this project [pytorch-pose-estimation](https://github.com/DavexPro/pytorch-pose-estimation)


# Quick-Start
1. Run pth_to_int.py to get the quantized model.
2. Run evaluate_model.py for inference.

# performance
1. Model size reduced from 200M to 50M.
2. Inference time is reduced by about 20%.
