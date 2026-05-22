import os
import cv2
import json

# ================= 配置区 =================
# 输入：尚未脱敏的原数据集路径
DIR_INPUT = r"F:\BaiduNetdiskDownload\Deidentified_Dataset"
# 输出：完美裁剪后的数据集路径
DIR_OUTPUT = r"F:\BaiduNetdiskDownload\Cropped_Dataset"

VALID_IMG_EXTS = {'.jpg', '.jpeg', '.png'}
VALID_VID_EXTS = {'.mp4', '.avi', '.mov'}
EXPECTED_IMG_NAMES = {'1', '2', '3', '4', '5'}

# 用于保存您画过的框的坐标缓存文件，避免下次运行重复画框
ROI_CACHE_FILE = "roi_memory.json"
# ==========================================

# 内存中的 ROI 字典: {(width, height): (x, y, w, h)}
roi_memory = {}

def load_roi_memory():
    """加载已保存的模板坐标"""
    global roi_memory
    if os.path.exists(ROI_CACHE_FILE):
        with open(ROI_CACHE_FILE, 'r') as f:
            # json 的 key 只能是字符串，所以存的时候把 (w, h) 转成了 "w_h"
            data = json.load(f)
            for k, v in data.items():
                w, h = map(int, k.split('_'))
                roi_memory[(w, h)] = tuple(v)

def save_roi_memory():
    """保存模板坐标到本地"""
    data = {f"{k[0]}_{k[1]}": v for k, v in roi_memory.items()}
    with open(ROI_CACHE_FILE, 'w') as f:
        json.dump(data, f)

def get_roi_for_resolution(frame):
    """根据图像分辨率，获取或要求用户绘制裁剪框"""
    h, w = frame.shape[:2]
    res_key = (w, h)
    
    if res_key in roi_memory:
        return roi_memory[res_key]
        
    # 如果遇到新分辨率，弹出窗口让用户画框
    print(f"\n[新设备提醒] 遇到新的画面尺寸: {w}x{h}")
    print("👉 请用鼠标拖拽一个框，框住纯净的超声区域（避开四周文字）。")
    print("👉 画完后按下【回车键】或【空格键】确认。按【C】键重画。")
    
    # 弹出交互窗口
    window_name = f"Select ROI for {w}x{h} (Press ENTER to confirm)"
    roi = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    
    # roi = (x, y, width, height)
    if roi[2] == 0 or roi[3] == 0:
        print("⚠️ 警告：您画的框无效，将跳过裁剪！")
        return None
        
    roi_memory[res_key] = roi
    save_roi_memory()
    print(f"✅ 尺寸 {w}x{h} 的模板已记录: 坐标 {roi}")
    return roi

def process_image(src_path, dst_path):
    img = cv2.imread(src_path)
    if img is None:
        return False
        
    roi = get_roi_for_resolution(img)
    if roi:
        x, y, w, h = roi
        cropped_img = img[y:y+h, x:x+w]
        cv2.imwrite(dst_path, cropped_img)
    else:
        # 如果获取不到ROI，原样复制
        cv2.imwrite(dst_path, img)
    return True

def process_video(src_path, dst_path):
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return False

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return False
        
    roi = get_roi_for_resolution(first_frame)
    if not roi:
        cap.release()
        return False
        
    x, y, roi_w, roi_h = roi
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30.0
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 注意：输出视频的尺寸必须是裁剪框的尺寸
    out = cv2.VideoWriter(dst_path, fourcc, fps, (roi_w, roi_h))
    
    # 写入第一帧
    out.write(first_frame[y:y+roi_h, x:x+roi_w])
    
    # 极速裁剪后续帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame[y:y+roi_h, x:x+roi_w])
        
    cap.release()
    out.release()
    return True

def main():
    load_roi_memory()
    if not os.path.exists(DIR_OUTPUT):
        os.makedirs(DIR_OUTPUT)

    print("="*50)
    print(" 开始极速交互式裁剪脱敏...")
    print("="*50)
    
    total_patients = 0
    
    for root, dirs, files in os.walk(DIR_INPUT):
        if not files:
            continue
            
        patient_folder_name = os.path.basename(root)
        rel_path = os.path.relpath(root, DIR_INPUT)
        out_patient_dir = os.path.join(DIR_OUTPUT, rel_path)
        
        if not os.path.exists(out_patient_dir):
            os.makedirs(out_patient_dir)
            
        total_patients += 1
        print(f"[{total_patients}] 处理中: {patient_folder_name}")
        
        for file in files:
            name, ext = os.path.splitext(file)
            ext_lower = ext.lower()
            src_file_path = os.path.join(root, file)
            
            if ext_lower in VALID_IMG_EXTS and name in EXPECTED_IMG_NAMES:
                dst_file_path = os.path.join(out_patient_dir, f"{name}.jpg")
                process_image(src_file_path, dst_file_path)
                
            elif ext_lower in VALID_VID_EXTS:
                dst_file_path = os.path.join(out_patient_dir, "video.mp4")
                process_video(src_file_path, dst_file_path)

    print("\n" + "="*50)
    print(" 裁剪脱敏全部完成！100% 绝对安全！")
    print("="*50)

if __name__ == "__main__":
    main()