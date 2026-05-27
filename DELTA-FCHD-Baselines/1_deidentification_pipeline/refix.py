import os
import json
import cv2

DIR_DATASET = r"F:\BaiduNetdiskDownload\Final_Clean_Dataset"
DIR_QA = r"F:\BaiduNetdiskDownload\QA_Review"

def apply_blackout(img, shapes):
    """Draw black masks according to AnyLabeling coordinates."""
    for shape in shapes:
        points = shape['points']
        # AnyLabeling points format: [[x1, y1], [x2, y2]]
        pt1 = (int(points[0][0]), int(points[0][1]))
        pt2 = (int(points[1][0]), int(points[1][1]))
        cv2.rectangle(img, pt1, pt2, (0,0,0), -1)
    return img

# Iterate over all JSON files under QA_Review
for file in os.listdir(DIR_QA):
    if not file.endswith('.json'): continue
    
    with open(os.path.join(DIR_QA, file), 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    image_path = data['imagePath']  # e.g., Video_C01_0033__video.jpg
    shapes = data['shapes']
    if not shapes: continue
    
    # Recover the original patient path
    # Parse Video_C01_0033__video.jpg -> folder Video_C01_0033, file video.mp4
    parts = image_path.replace('.jpg', '').split('__')
    patient_id = parts[0]
    file_type = parts[1]  # '1', '2', or 'video'
    
    # Assume the patient folder is under the corresponding modality; use os.walk if not found
    # For simplicity, use os.walk to find the corresponding patient folder in the dataset
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
        
        # Replace the original video
        os.remove(src_video)
        os.rename(temp_video, src_video)
        print(f"Video fixed successfully: {patient_id}")
        
    else:
        # Process static image
        src_img = os.path.join(target_folder, f"{file_type}.jpg")
        img = cv2.imread(src_img)
        img = apply_blackout(img, shapes)
        cv2.imwrite(src_img, img)
        print(f"Image fixed successfully: {patient_id} / {file_type}.jpg")

print("All missed cases have been fixed.")
