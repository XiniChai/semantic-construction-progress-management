# train_yolo_shell.py

from ultralytics import YOLO

def main():
    # === 配置路径 ===
    dataset_yaml_path = r"C:\Users\WUMU\ultralytics-main\ultralytics\datasets\datasets.yaml"
    pretrained_model = "yolov8s.pt"  # 你也可以换成 yolov8s.pt 等
    output_name = "shell_detector"   # 输出实验名

    # === 创建模型并开始训练 ===
    model = YOLO(pretrained_model)

    model.train(
        data=dataset_yaml_path,
        workers=8,  # 数据集配置文件
        epochs=400,
        imgsz=(544,960),
        batch=16,
        name=output_name,
        project="runs/detect/0826/view_2",  # 默认路径为 runs/detect/shell_detector
        verbose=True,

        # === 学习率相关 ===
        lr0=0.0004, # 初始学习率（建议降低）
        lrf=0.01, # 最终学习率比例
        warmup_epochs=8, # 热启动轮次
        cos_lr=True, # Cosine 余弦退火学习率调度器


        # === 数据增强参数 ===
        degrees=5.0,            # 小幅旋转
        translate=0.05,          # 平移
        scale=0.1,              # 缩放
        shear=0.1,             # 裁剪角度变换，初期可以忽略
        perspective=0.001,    #透视扭曲变换概率
        flipud=0.2,            #垂直翻转概率
        fliplr=0.5,             # 水平翻转（常用且有效）
        hsv_h=0.015,            # 色调调整
        hsv_s=0.7,              # 饱和度调整
        hsv_v=0.4,              # 亮度调整
        mosaic=0.5,             # Mosaic 增强（YOLO 特有，强烈推荐）
        mixup=0.2,             # MixUp 图像融合增强概率,不稳定，需更多数据时再考虑
        copy_paste=0.2,         #是否使用 copy-paste 增强

        # === 自定义损失函数权重（见下文说明）===
        box=0.1, # 框回归损失权重较低（不压过其它）
        cls=0.8, # 分类损失权重增强（提升精度）
        dfl=1.5, # 分布式回归损失（提高位置回归质量）

        # === 提前停止（防止过拟合）===
        patience=30 # 验证集指标超过50轮无提升则停止 
    )

    print("\n✅ 训练完成，模型保存在 runs/detect/0826/view_2/shell_detector/weights/best.pt")

# ==== 关键：仅在主线程中运行 ====
if __name__ == "__main__":
    main()

