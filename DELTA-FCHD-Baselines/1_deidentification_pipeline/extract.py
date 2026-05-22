import os
import cv2
import shutil

DIR_DATASET = r"F:\BaiduNetdiskDownload\Final_Clean_Dataset"
DIR_QA = r"F:\BaiduNetdiskDownload\QA_Review"

if not os.path.exists(DIR_QA):
    os.makedirs(DIR_QA)

for root, dirs, files in os.walk(DIR_DATASET):
    if not files: continue
    
    patient_id = os.path.basename(root)
    for file in files:
        src_path = os.path.join(root, file)
        name, ext = os.path.splitext(file)
        
        # 静态图直接拷贝并重命名
        if ext.lower() in {'.jpg', '.png'}:
            dst_name = f"{patient_id}__{name}.jpg"
            shutil.copy2(src_path, os.path.join(DIR_QA, dst_name))
            
        # 视频只抽取第一帧
        elif ext.lower() in {'.mp4', '.avi'}:
            cap = cv2.VideoCapture(src_path)
            ret, frame = cap.read()
            if ret:
                dst_name = f"{patient_id}__video.jpg"
                cv2.imwrite(os.path.join(DIR_QA, dst_name), frame)
            cap.release()

print("抽帧完毕！请前往 QA_Review 文件夹进行极速筛查。")