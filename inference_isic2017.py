import os
import torch
import numpy as np
from collections import defaultdict, OrderedDict
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm
import cv2

from dataset_isic2017 import ISICDataset
from model import build_model


# -------------------------------------------------------------
#  Dice / IoU 计算（适配二分类 ISIC 任务）
# -------------------------------------------------------------
# def calculate_metrics(pred, label):
#     eps = 1e-6
#
#     pred_bin = (pred > 0).astype(np.uint8)
#     label_bin = (label > 0).astype(np.uint8)
#
#     TP = np.logical_and(pred_bin == 1, label_bin == 1).sum()
#     FP = np.logical_and(pred_bin == 1, label_bin == 0).sum()
#     FN = np.logical_and(pred_bin == 0, label_bin == 1).sum()
#
#     dice = (2 * TP + eps) / (2 * TP + FP + FN + eps)
#     iou = (TP + eps) / (TP + FP + FN + eps)
#
#     return {"Dice": dice, "IoU": iou}
def calculate_metrics(pred, label):
    eps = 1e-6

    pred_bin = (pred > 0).astype(np.uint8)
    label_bin = (label > 0).astype(np.uint8)

    TP = np.logical_and(pred_bin == 1, label_bin == 1).sum()
    TN = np.logical_and(pred_bin == 0, label_bin == 0).sum()
    FP = np.logical_and(pred_bin == 1, label_bin == 0).sum()
    FN = np.logical_and(pred_bin == 0, label_bin == 1).sum()

    # Dice
    dice = (2 * TP + eps) / (2 * TP + FP + FN + eps)

    # IoU
    iou = (TP + eps) / (TP + FP + FN + eps)

    # Accuracy
    acc = (TP + TN + eps) / (TP + TN + FP + FN + eps)

    # Sensitivity (Recall)
    sen = (TP + eps) / (TP + FN + eps)

    # Specificity
    spe = (TN + eps) / (TN + FP + eps)

    return {
        "Dice": dice,
        "IoU": iou,
        "Acc": acc,
        "Sen": sen,
        "Spe": spe,
    }


# -------------------------------------------------------------
#  单张图推理（Dataset 已保证尺寸一致，不再 resize）
# -------------------------------------------------------------
@torch.no_grad()
def eval_single_image(model, image, device):

    image = image.to(device)  # (1,3,H,W)
    output = model(image)     # (1,2,H,W)
    pred = torch.argmax(torch.softmax(output, dim=1), dim=1)  # (1,H,W)

    return pred.squeeze(0).cpu().numpy()


# -------------------------------------------------------------
#  保存预测图
# -------------------------------------------------------------
def save_prediction_png(pred, label, case_name, output_folder):

    pred_img = (pred * 255).astype(np.uint8)
    label_img = (label * 255).astype(np.uint8)

    cv2.imwrite(os.path.join(output_folder, f"{case_name}_pred.png"), pred_img)
    cv2.imwrite(os.path.join(output_folder, f"{case_name}_gt.png"), label_img)


# -------------------------------------------------------------
#  总入口
# -------------------------------------------------------------
def run_isic(
    ckpt_path: str,
    dataset_dir: str = "dataset/isic2017/val",
    output_folder: str = "testing_isic2017",
    num_classes: int = 2,
    img_size: int = 256,
):

    os.makedirs(output_folder, exist_ok=True)
    logger.add(os.path.join(output_folder, "test.log"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Device] {device}")

    # ------------------- 加载模型 -------------------
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]

    # 去掉 Lightning 的 _model.xx 前缀
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        new_state_dict[k.replace("_model.", "")] = v

    model = build_model(in_channels=3, num_classes=num_classes)
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    # ------------------- 数据集 -------------------
    dataset = ISICDataset(
        root=dataset_dir,
        img_size=img_size,
        split='val',
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    logger.info(f"Start testing {len(dataset)} samples.")

    # ------------------- 推理 -------------------
    all_metrics = defaultdict(list)

    for sample in tqdm(dataloader):
        image = sample["image"]         # (1,3,H,W)
        label = sample["label"].numpy()  # (1,H,W)
        case_name = sample["case_name"][0]

        pred = eval_single_image(model, image, device)
        label_np = label.squeeze(0)

        metrics = calculate_metrics(pred, label_np)
        for k, v in metrics.items():
            all_metrics[k].append(v)

        save_prediction_png(pred, label_np, case_name, output_folder)

    # ------------------- 最终指标 -------------------
    final_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    logger.info("=== Final Results ===")
    for k, v in final_metrics.items():
        logger.info(f"{k}: {v:.4f}")

    return final_metrics


def main():
    run_isic(
        ckpt_path="log/msvm-unet-isic/checkpoints/epoch=189-val_mean_dice=0.8891.ckpt",
        dataset_dir="dataset/isic2017/val",
        output_folder="testing_isic2017",
        num_classes=2,
        img_size=256,
    )


if __name__ == "__main__":
    main()
