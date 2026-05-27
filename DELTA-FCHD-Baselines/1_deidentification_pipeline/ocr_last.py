import os
import cv2
import json
import easyocr
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# ================= Configuration =================
# Note: input should be the original de-identified dataset before cropping
DIR_INPUT = r"F:\BaiduNetdiskDownload\Deidentified_Dataset"
# Output: final dataset path after two-stage de-identification
DIR_OUTPUT = r"F:\BaiduNetdiskDownload\Final_Clean_Dataset"

# Cache file for previously selected ROI boxes
ROI_CACHE_FILE = "roi_memory.json"

VALID_IMG_EXTS = {'.jpg', '.jpeg', '.png'}
VALID_VID_EXTS = {'.mp4', '.avi', '.mov'}
EXPECTED_IMG_NAMES = {'1', '2', '3', '4', '5'}

# Conservative OCR parameters
CONFIDENCE_THRESHOLD = 0.60
PADDING = 2
# ==========================================

print("Loading conservative OCR model...")
reader = easyocr.Reader(['en', 'ch_sim'], gpu=True)
print("Model loaded.\n")

roi_memory = {}

def load_roi_memory():
    """Load ROI templates and force even dimensions."""
    if os.path.exists(ROI_CACHE_FILE):
        with open(ROI_CACHE_FILE, 'r') as f:
            data = json.load(f)
            for k, v in data.items():
                w, h = map(int, k.split('_'))
                x, y, roi_w, roi_h = v
                
                # Core fix: force width and height to be even to avoid encoder failures.
                if roi_w % 2 != 0: roi_w -= 1
                if roi_h % 2 != 0: roi_h -= 1
                    
                roi_memory[(w, h)] = (x, y, roi_w, roi_h)

def save_roi_memory():
    data = {f"{k[0]}_{k[1]}": v for k, v in roi_memory.items()}
    with open(ROI_CACHE_FILE, 'w') as f:
        json.dump(data, f)

current_roi = None 
def onselect(eclick, erelease):
    global current_roi
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    x, y = min(x1, x2), min(y1, y2)
    w, h = abs(x2 - x1), abs(y2 - y1)
    
    # Force manually selected ROI dimensions to be even as well
    if w % 2 != 0: w -= 1
    if h % 2 != 0: h -= 1
        
    current_roi = (x, y, w, h)
    print(f"  --> Selected ROI: {current_roi}. Please close the image window to continue.")

def get_roi_for_resolution(frame):
    global current_roi
    h, w = frame.shape[:2]
    res_key = (w, h)
    
    if res_key in roi_memory:
        return roi_memory[res_key]
        
    print(f"\n[New resolution] {w}x{h}. Select the ROI in the pop-up window, then close it.")
    current_roi = None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(frame_rgb)
    ax.set_title(f"Resolution: {w}x{h} | Draw rectangle, then CLOSE window")
    rs = RectangleSelector(ax, onselect, useblit=True, button=[1], interactive=True)
    plt.show() 
    
    if current_roi is None or current_roi[2] == 0 or current_roi[3] == 0:
        return None
        
    roi_memory[res_key] = current_roi
    save_roi_memory()
    return current_roi

def get_conservative_redaction_boxes(frame):
    results = reader.readtext(frame, detail=1)
    boxes = []
    h, w = frame.shape[:2]
    for (bbox, text, prob) in results:
        if prob >= CONFIDENCE_THRESHOLD:
            tl_x = max(0, int(bbox[0][0]) - PADDING)
            tl_y = max(0, int(bbox[0][1]) - PADDING)
            br_x = min(w, int(bbox[2][0]) + PADDING)
            br_y = min(h, int(bbox[2][1]) + PADDING)
            boxes.append(((tl_x, tl_y), (br_x, br_y)))
            print(f"      [Residual text redaction] '{text}'")
    return boxes

def apply_redaction(frame, boxes):
    for (pt1, pt2) in boxes:
        cv2.rectangle(frame, pt1, pt2, (0, 0, 0), thickness=-1)
    return frame

def process_image(src_path, dst_path):
    img = cv2.imread(src_path)
    if img is None: return False
        
    roi = get_roi_for_resolution(img)
    if roi:
        x, y, w, h = roi
        img = img[y:y+h, x:x+w]  # Apply cropping
        
    boxes = get_conservative_redaction_boxes(img)
    if boxes:
        img = apply_redaction(img, boxes)  # Apply OCR redaction
        
    cv2.imwrite(dst_path, img)
    return True

def process_video(src_path, dst_path):
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened(): return False

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return False
        
    roi = get_roi_for_resolution(first_frame)
    if not roi:
        cap.release()
        return False
        
    x, y, roi_w, roi_h = roi
    
    # ================= Frame-rate correction =================
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps or fps > 120 or fps < 1:
        fps = 30.0
    else:
        fps = float(round(fps))  # Round unusual fractional frame rates
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(dst_path, fourcc, fps, (roi_w, roi_h))
    
    cropped_first = first_frame[y:y+roi_h, x:x+roi_w]
    redaction_boxes = get_conservative_redaction_boxes(cropped_first)
    
    if redaction_boxes:
        out.write(apply_redaction(cropped_first, redaction_boxes))
        while True:
            ret, frame = cap.read()
            if not ret: break
            cropped = frame[y:y+roi_h, x:x+roi_w]
            out.write(apply_redaction(cropped, redaction_boxes))
    else:
        out.write(cropped_first)
        while True:
            ret, frame = cap.read()
            if not ret: break
            cropped = frame[y:y+roi_h, x:x+roi_w]
            out.write(cropped)
            
    cap.release()
    out.release()
    return True

def main():
    load_roi_memory()
    if not os.path.exists(DIR_OUTPUT):
        os.makedirs(DIR_OUTPUT)

    print("="*60)
    print("Starting final pipeline: cropping and OCR redaction...")
    print("="*60)
    
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
        print(f"\n[{total_patients}] Processing: {patient_folder_name}")
        
        for file in files:
            name, ext = os.path.splitext(file)
            ext_lower = ext.lower()
            src_file_path = os.path.join(root, file)
            dst_file_path = os.path.join(out_patient_dir, file)
            
            if ext_lower in VALID_IMG_EXTS:
                process_image(src_file_path, dst_file_path)
            elif ext_lower in VALID_VID_EXTS:
                process_video(src_file_path, dst_file_path)

    print("\n" + "="*60)
    print("Two-stage de-identification pipeline completed.")
    print("="*60)

if __name__ == "__main__":
    main()
