import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gc
import logging
import glob
import numpy as np
from PIL import Image
from skimage import io
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import RescaleT, ToTensorLab, SalObjDataset
from model import BASNet

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normPRED(d):
    """归一化预测结果"""
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

def save_output(image_name, pred, d_dir, sub_dir):
    """保存预测结果"""
    # 确保输出目录存在
    output_dir = os.path.join(d_dir, sub_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        # logging.info(f"创建目录：{output_dir}")

    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = os.path.basename(image_name)  # 直接获取文件名
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    output_path = os.path.join(output_dir, img_name)  # 自动对应目录名称
    imo.save(output_path)

class BASNetRunner:
    """常驻内存的 BASNet 推理器，可跨多批图片复用模型。"""

    def __init__(self, model_dir):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"模型文件未找到：{model_dir}")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logging.info(f"加载 BASNet 到设备: {self.device}")
        self.net = BASNet(3, 1).to(self.device)
        self.net.load_state_dict(torch.load(model_dir, map_location=self.device))
        self.net.eval()

    def run(self, image_dir, prediction_dir):
        img_name_list = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        if not img_name_list:
            logging.warning(f"未找到图像文件：{image_dir}")
            return
        sub_dir = os.path.basename(image_dir)
        dataset = SalObjDataset(
            img_name_list=img_name_list,
            lbl_name_list=[],
            transform=transforms.Compose([RescaleT(256), ToTensorLab(flag=0)]),
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        try:
            with torch.inference_mode():
                for image_name, data in zip(img_name_list, dataloader):
                    inputs = data['image'].float().to(self.device)
                    outputs = self.net(inputs)
                    pred = normPRED(outputs[0][:, 0, :, :])
                    save_output(image_name, pred, prediction_dir, sub_dir)
                    del inputs, outputs, pred
        finally:
            del dataloader, dataset
            gc.collect()

    def close(self):
        if self.net is None:
            return
        del self.net
        self.net = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def run_basnet_test(image_dir, prediction_dir, model_dir):
    """兼容原调用方式：单次加载、运行并释放 BASNet。"""
    with BASNetRunner(model_dir) as runner:
        runner.run(image_dir, prediction_dir)

if __name__ == "__main__":
    # 获取当前脚本的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 获取上一级目录
    parent_dir = os.path.dirname(current_dir)
    
    # 定义路径（修复路径拼接问题）
    image_dir = os.path.join(parent_dir, "umap_digit_min", "umap_plots_digits_colors")
    prediction_dir = os.path.join(parent_dir, "umap_digit_min", "umap_results")
    model_dir = os.path.join(current_dir, "saved_models", "basnet_bsi", "basnet_best_train_massvis.pth")
    
    # 运行测试
    run_basnet_test(image_dir, prediction_dir, model_dir)
