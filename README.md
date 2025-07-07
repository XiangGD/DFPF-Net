# DFPF-Net
This repo holds code for [Image Manipulation Localization Using Dual-Shallow Feature Pyramid Fusion and Boundary Contextual Incoherence Enhancement](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10771742)

## Usage

### 1. Download Google pre-trained ViT models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../models/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare data

Please download the IML-MUST dataset.<br>
* [Baidu Disk](https://pan.baidu.com/s/180TzwbTHj1Q3FOvIwT3vyg?pwd=gdit) <br>
* [Google Drive](https://drive.google.com/drive/folders/1bCCRaP7MKkEhxbTBbcKvy0AHBFi6ZMQQ?usp=drive_link)

### 3. Environment

Please prepare an environment with Python=3.8, Pytorch=1.10.1, and then use the command "pip install -r requirements.txt" for the dependencies.

### 4. Train/Test

- Run the train script on the IML-MUST dataset. The batch size can be reduced to 18 or 21 to save memory, and both can reach similar performance.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset IML-MUST --vit_name R50-ViT-B_16
```

- Run the test script on the Coverage dataset.

```bash
python test.py --dataset Coverage --vit_name R50-ViT-B_16
```

### 5. CKPT
* [Google Dirve](https://drive.google.com/drive/folders/1FvU6Q7U_XLMO8At4f141HZt6jWeg1f27?usp=drive_link)
* [Baudu Disk](https://pan.baidu.com/s/1q9-TXOtGL6ZtHAl1zWKJwQ?pwd=gdit)

## Citations

```bibtex
@ARTICLE{10771742,
  author={Xiang, Yan and Yuan, Xiaochen and Zhao, Kaiqi and Liu, Tong and Xie, Zhiyao and Huang, Guoheng and Li, Jianqing},
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence}, 
  title={Image Manipulation Localization Using Dual-Shallow Feature Pyramid Fusion and Boundary Contextual Incoherence Enhancement}, 
  year={2024},
  volume={},
  number={},
  pages={1-11},
  doi={10.1109/TETCI.2024.3500025}}
```
