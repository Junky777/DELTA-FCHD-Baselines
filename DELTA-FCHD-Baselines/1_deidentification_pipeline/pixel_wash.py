import os
import cv2
import json

# ================= Configuration =================
# Input: raw dataset path before de-identification
DIR_INPUT = r"F:\BaiduNetdiskDownload\Deidentified_Dataset"
# Output: cropped dataset path
DIR_OUTPUT = r"F:\BaiduNetdiskDownload\Cropped_Dataset"

VALID_IMG_EXTS = {'.jpg', '.jpeg', '.png'}
VALID_VID_EXTS = {'.mp4', '.avi', '.mov'}
EXPECTED_IMG_NAMES = {'1', '2', '3', '4', '5'}

# Cache file for storing previously selected ROI boxes
ROI_CACHE_FILE = "roi_memory.json"
# ==========================================

# In-memory ROI dictionary: {(width, height): (x, y, w, h)}
roi_memory = {}

def load_roi_memory():
    """Load saved ROI template coordinates."""
    global roi_memory
    if os.path.exists(ROI_CACHE_FILE):
        with open(ROI_CACHE_FILE, 'r') as f:
            # JSON keys must be strings, so (w, h) is stored as "w_h".
            data = json.load(f)
            for k, v in data.items():
                w, h = map(int, k.split('_'))
                roi_memory[(w, h)] = tuple(v)

def save_roi_memory():
    """Save ROI template coordinates locally."""
    data = {f"{k[0]}_{k[1]}": v for k, v in roi_memory.items()}
    with open(ROI_CACHE_FILE, 'w') as f:
        json.dump(data, f)

def get_roi_for_resolution(frame):
    """Get or request an ROI crop box according to image resolution."""
    h, w = frame.shape[:2]
    res_key = (w, h)
    
    if res_key in roi_memory:
        return roi_memory[res_key]
        
    # If a new resolution is encountered, open a window for ROI selection
    print(f"\n[New resolution] New frame size detected: {w}x{h}")
    print("Drag the mouse to select the clean ultrasound region, avoiding surrounding text.")
    print("Press Enter or Space to confirm. Press C to redraw.")
    
    # Open interactive window
    window_name = f"Select ROI for {w}x{h} (Press ENTER to confirm)"
    roi = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    
    # roi = (x, y, width, height)
    if roi[2] == 0 or roi[3] == 0:
        print("Warning: invalid ROI box; cropping will be skipped.")
        return None
        
    roi_memory[res_key] = roi
    save_roi_memory()
    print(f"Resolution {w}x{h} template recorded: coordinates {roi}")
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
        # If no ROI is available, copy the file unchanged
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
    # Note: output video size must match the crop box size
    out = cv2.VideoWriter(dst_path, fourcc, fps, (roi_w, roi_h))
    
    # Write the first frame
    out.write(first_frame[y:y+roi_h, x:x+roi_w])
    
    # Crop subsequent frames
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
    print("Starting interactive ROI cropping for de-identification...")
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
        print(f"[{total_patients}] Processing: {patient_folder_name}")
        
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
    print("Cropping-based de-identification completed.")
    print("="*50)

if __name__ == "__main__":
    main()
