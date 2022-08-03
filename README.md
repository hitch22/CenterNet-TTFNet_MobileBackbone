#  CenterNet-TTFNet_MobileBackbone
Tensorflow2 and Keras implementation of CenterNet and TTFNet

## Update
1. [22/07/2] Update: Concat Path Aggregation in FPN and Simple Neck UPUP

## Performance
All models are trained at coco 2017 train 118k and evaluated at coco 2017 val 5k

Model | Lr schedule  | max learning rate | BatchSize | total epochs | kernel regulaization | optimizer | Loss | Input Size | Training Precision | Params[M] | FLOPs[G] | mAP 0.5:0.95@0.05 |
| ------------------------------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
MobileNetV3Large FPNLite TTFNet | CosineDecay with warmup | 5e-3 | 64 | 300 | 3e-5 | Gradient Centralization SGDM | Focal, DIOU |320x320| FP16 | 1.8 | 1.3 | 23.1 |
MobileNetDet FPNLite TTFNet | CosineDecay with warmup | 5e-3 | 64 | 300 | 3e-5 | Gradient Centralization SGDM | Focal, DIOU |320x320| FP16 | 3.1 | 1.4 | 23.6 |

## Inference Examples
<img width="49%" src="https://user-images.githubusercontent.com/89026839/179469473-ab768561-0964-4ec7-ad0b-ebf0c1033570.png"/> <img width="49%" src="https://user-images.githubusercontent.com/89026839/179469556-f5801692-9c5b-43b8-8837-2645b2b2a152.png"/>

## Reference
1. Searching for MobileNetV3 https://arxiv.org/abs/1905.02244

2. MobileDets: Searching for Object Detection Architectures for Mobile Accelerators https://arxiv.org/abs/2004.14525

3. Objects as Points https://arxiv.org/abs/1904.07850

4. Training-Time-Friendly Network for Real-Time Object Detection https://arxiv.org/abs/1909.00700

5. Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression https://arxiv.org/abs/1911.08287

