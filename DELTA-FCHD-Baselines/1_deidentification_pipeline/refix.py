import os
import json
import cv2

DIR_DATASET = r"F:\BaiduNetdiskDownload\Final_Clean_Dataset"
DIR_QA = r"F:\BaiduNetdiskDownload\QA_Review"

def apply_blackout(img, shapes):
    """根据 AnyLabeling 的坐标画黑块"""
    for shape in shapes:
        points = shape['points']
        # AnyLabeling points format: [[x1, y1], [x2, y2]]
        pt1 = (int(points[0][0]), int(points[0][1]))
        pt2 = (int(points[1][0]), int(points[1][1]))
        cv2.rectangle(img, pt1, pt2, (0,0,0), -1)
    return img

# 遍历 QA_Review 下所有的 json 文件
for file in os.listdir(DIR_QA):
    if not file.endswith('.json'): continue
    
    with open(os.path.join(DIR_QA, file), 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    image_path = data['imagePath'] # 如: Video_C01_0033__video.jpg
    shapes = data['shapes']
    if not shapes: continue
    
    # 还原出原始病人的真实路径
    # 解析 Video_C01_0033__video.jpg -> 文件夹 Video_C01_0033, 文件 video.mp4
    parts = image_path.replace('.jpg', '').split('__')
    patient_id = parts[0]
    file_type = parts[1] # '1', '2' 或 'video'
    
    # 这里假设病人文件夹在对应模态下，如果找不到可以使用 os.walk 去找
    # 简单起见，利用 os.walk 在数据集中寻找对应的病人文件夹
    target_folder = None
    for root, dirs, files in os.walk(DIR_DATASET):
        if os.path.basename(root) == patient_id:
            target_folder = root
            break
            
    if not target_folder: continue
    
    if file_type == 'video':
        src_video = os.path.join(target_folder, "video.mp4")
        temp_video = os.path.join(target_folder, "video_temp.mp4")
        
        cap = cv2.VideoCapture(src_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        while True:
            ret, frame = cap.read()
            if not ret: break
            out.write(apply_blackout(frame, shapes))
        cap.release()
        out.release()
        
        # 替换原视频
        os.remove(src_video)
        os.rename(temp_video, src_video)
        print(f"✅ 成功修复视频: {patient_id}")
        
    else:
        # 处理静态图
        src_img = os.path.join(target_folder, f"{file_type}.jpg")
        img = cv2.imread(src_img)
        img = apply_blackout(img, shapes)
        cv2.imwrite(src_img, img)
        print(f"✅ 成功修复图像: {patient_id} / {file_type}.jpg")

print("所有漏网之鱼修复完毕！")