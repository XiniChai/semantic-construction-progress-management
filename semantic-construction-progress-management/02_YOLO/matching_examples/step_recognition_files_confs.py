import os
import glob
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ==================================================
# CONFIG
# ==================================================

MODEL_PATH = r"C:/Users/Xini Chai/ultralytics-main/runs/detect/best_v1_0902_4.pt" #best_v1_0902_4.pt
INPUT_FOLDER = r"D:\Master SEU\02-thesis\05-cooperation thesis\05-Construction Semantic web\Dataset\real_images\view_1\test_1"
STEP_VIEW_INDEX_JSON = r"C:\Users\10205\step_view_index_v1.json"
OUTPUT_FOLDER = r"C:\Users\Xini Chai\ultralytics-main\runs\detect\00_test\view_1_0707"


# YOLO参数
CONF_THRES = 0.45 #0.25
IOU_NMS = 0.45

# IoU匹配阈值
MATCH_IOU_THRES = 0.25


# BIM导图尺寸
ORIG_W = 1280
ORIG_H = 720

RESIZE_W = 960
RESIZE_H = 540

FINAL_W = 960
FINAL_H = 544

PAD_TOP = 2


# ==================================================
# 图像预处理
# ==================================================

def preprocess_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(image_path)

    resized = cv2.resize(
        img,
        (RESIZE_W, RESIZE_H)
    )

    letterboxed = cv2.copyMakeBorder(
        resized,
        2,
        2,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    return letterboxed


# ==================================================
# bbox转换
# ==================================================

def transform_bbox(bbox):

    x1, y1, x2, y2 = bbox

    sx = RESIZE_W / ORIG_W
    sy = RESIZE_H / ORIG_H

    return [
        int(round(x1 * sx)),
        int(round(y1 * sy + PAD_TOP)),
        int(round(x2 * sx)),
        int(round(y2 * sy + PAD_TOP))
    ]


# ==================================================
# IoU
# ==================================================

def iou_xyxy(boxA, boxB):

    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    xA = max(ax1, bx1)
    yA = max(ay1, by1)
    xB = min(ax2, bx2)
    yB = min(ay2, by2)

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)

    interArea = interW * interH

    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = areaA + areaB - interArea + 1e-6

    return interArea / union


# ==================================================
# 读取JSON
# ==================================================

def load_step_index(json_path):

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data["frames"]

    return data


# ==================================================
# 建立Step数据库
# ==================================================

def build_step_database(frames):

    step_db = {}

    for frame in frames:

        step = frame["step"]
        camera = frame["camera_index"]

        key = (step, camera)

        comps = []

        for c in frame["components"]:

            comps.append({

                "id": c["id"],

                "bbox": transform_bbox(
                    c["bbox_2d"]
                )

            })

        step_db[key] = comps

        #step_db.setdefault(
        #   step,
        #     []
        #).extend(comps)


    #====增加验证=====
    for step in ["P1_01","P1_04"]:

        print(
            step,
            len(step_db[key])
        )
    #====增加验证=====

    return step_db



# ==================================================
# Step识别
# ==================================================

def pick_best_step(pred_boxes, step_db):

    best_step = None
    best_camera = None
    best_score = -1

    #for step, comps in step_db.items():

    for key, comps in step_db.items():

        step,camera = key

        score = 0

        for pb in pred_boxes:

            max_iou = 0

            for comp in comps:

                iou = iou_xyxy(
                    pb,
                    comp["bbox"]
                )

                if iou > max_iou:
                    max_iou = iou

            score += max_iou

        if score > best_score:

            best_score = score
            best_step = step
            best_camera = camera

    return best_step, best_camera


# ==================================================
# 构件匹配
# ==================================================

def match_components(pred_boxes, step_components):

    match_info = {}

    used = set()

    for i, pb in enumerate(pred_boxes):

        #=======增加调试IoU矩阵========
        print("\n--------------------")
        print(f"Prediction Box {i}")
        #=======增加调试IoU矩阵========

        best_iou = 0
        best_id = None
        best_j = None

        for j, comp in enumerate(step_components):

            if j in used:
                continue

            iou = iou_xyxy(
                pb,
                comp["bbox"]
            )

            #===增加内容===
            print(
                f"{comp['id']} "
                f"IoU={iou:.3f}"
            )
            #===增加内容===

            if iou > best_iou:

                best_iou = iou
                best_id = comp["id"]
                best_j = j

        if best_iou >= MATCH_IOU_THRES:

            match_info[i] = {

                "component_id": best_id,

                "iou": best_iou

            }

            used.add(best_j)


        #增加打印验证
        print(
            f"Pred Box {i}"
        )

        print(
            f"Best ID={best_id}"
        )

        print(
            f"Best IoU={best_iou:.3f}"
        )
         #增加打印验证

         
    return match_info


# ==================================================
# 可视化
# ==================================================

def draw_result(image, pred_boxes, pred_confs, match_info, step):

    vis = image.copy()

    ious = []

    component_ids = []

    for i, box in enumerate(pred_boxes):

        x1, y1, x2, y2 = box
        
        conf = pred_confs[i]

        if i in match_info:

            cid = match_info[i]["component_id"]
            iou = match_info[i]["iou"]
            component_ids.append(cid)
            ious.append(iou)
            
            label = (
                f"{cid} "
                f"| IoU:{iou:.2f} "
                f"| Conf:{conf:.2f}"
            )

            color = (0, 0, 255)

        else:

            label = (
                f"Unmatched "
                f"| Conf:{conf:.2f}"
            )

            color = (0, 255, 0)

        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            vis,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    avg_iou = np.mean(ious) if len(ious) > 0 else 0
    avg_conf = np.mean(pred_confs)

    component_ids = sorted(
        list(set(component_ids))
    )

    component_text = ",".join(
        component_ids[:5]
    )

    if len(component_ids) > 5:
        component_text += "..."

    cv2.putText(
        vis,
        f"Predicted Process : {step}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        vis,
        f"Components : {component_text}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.putText(
        vis,
        f"Mean Max-IoU : {avg_iou:.2f}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        vis,
        f"Mean Confidence : {avg_conf:.2f}",
        (20,160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,0,0),
        2
    )


    print(
        "Matched IDs:",
        component_ids
    )

    return vis


# ==================================================
# 单张图处理
# ==================================================

def process_one_image(image_path, model, step_db):

    image = preprocess_image(image_path)

    #results = model.predict(
    #   source=image,
    #    imgsz=1280,
    #    conf=0.05, #0.25
    #    iou=0.90    #0.45
    #)

    results = model.predict(
        source=image,
        conf=CONF_THRES,
        iou=IOU_NMS,
        imgsz=(544,960),
        verbose=False
    )

    pred_boxes = []
    pred_confs = []

    for r in results:

        if r.boxes is None:
            continue

        for b in r.boxes:

            x1, y1, x2, y2 = map(
                int,
                b.xyxy[0].tolist()
            )


            conf = float(
                b.conf[0]
            )

            pred_boxes.append(
                [x1, y1, x2, y2]
            )


            pred_confs.append(
                conf
            )

            #====增加====
            print(
                b.xyxy[0].tolist(),
                float(b.conf[0])
            )
            #====增加====

    #=====增加YOLO检测结果======
    print("\n========================")
    print(Path(image_path).name)

    print(
        f"YOLO Detections = {len(pred_boxes)}"
    )

    for i, conf in enumerate(pred_confs):

        print(
            f"Box {i} "
            f"Conf={conf:.3f}"
        )
    #==========================

    if len(pred_boxes) == 0:

        print(f"No detection: {os.path.basename(image_path)}")

        return

    best_step,best_camera = pick_best_step(
        pred_boxes,
        step_db
    )

    step_components = step_db[
        (best_step, best_camera)
    ]

    match_info = match_components(
        pred_boxes,
        step_components
    )

    #====验证构件识别数量===
    print("\nMatch Info")

    for k,v in match_info.items():

        print(
            k,
            v["component_id"],
            v["iou"]
        )
    #====验证构件识别数量===

    vis = draw_result(
        image,
        pred_boxes,
        pred_confs,
        match_info,
        best_step
    )

    save_path = os.path.join(
        OUTPUT_FOLDER,
        Path(image_path).stem + "_result.jpg"
    )

    cv2.imwrite(
        save_path,
        vis
    )

    print(
        f"{Path(image_path).name}"
        f" --> Step={best_step}"
        f" | Match={len(match_info)}"
    )


# ==================================================
# MAIN
# ==================================================

def main():

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    print("Loading YOLO...")

    model = YOLO(MODEL_PATH)
    frames = load_step_index(
        STEP_VIEW_INDEX_JSON
    )

    step_db = build_step_database(
        frames
    )

    image_list = []
    image_list.extend(
        glob.glob(
            os.path.join(
                INPUT_FOLDER,
                "*.jpg"
            )
        )
    )

    image_list.extend(
        glob.glob(
            os.path.join(
                INPUT_FOLDER,
                "*.png"
            )
        )
    )

    print(
        f"Found {len(image_list)} images"
    )

    for image_path in image_list:

        process_one_image(
            image_path,
            model,
            step_db
        )

    print("Finished.")


if __name__ == "__main__":
    main()