import cv2
import tkinter as tk
from tkinter import filedialog
import os
import time
from ultralytics import YOLO
import json
import numpy as np

# TKinter GUI 숨기기
root = tk.Tk()
root.withdraw()

# 결과 저장 폴더 선택
output_folder = filedialog.askdirectory(title='결과 저장 폴더를 선택하세요')
if not output_folder:
    print("결과를 저장할 폴더를 선택하지 않았습니다. 프로그램을 종료합니다.")
    exit()

# 저장용 폴더 생성
video_path = os.path.join(output_folder, 'recorded_video.avi')
image_output_folder = os.path.join(output_folder, 'image')
json_output_folder = os.path.join(output_folder, 'json')
os.makedirs(image_output_folder, exist_ok=True)
os.makedirs(json_output_folder, exist_ok=True)
mp4_output = os.path.join(output_folder, 'user_output_side.mp4')

# 기준 pose json 로딩 함수
def load_reference_pose(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    keypoints = []
    for kp in data['keypoints']:
        if kp["x"] is not None and kp["y"] is not None:
            keypoints.append([kp["x"], kp["y"]])
        else:
            keypoints.append([None, None])
    return np.array(keypoints, dtype=np.float32)

# 정확도 기반 유사성 판단 함수 (절대 거리 오차 20px 이하 기준 70% 이상 일치)
def is_pose_similar_by_accuracy(current_pose, reference_pose, threshold_px=20, ratio=0.7):
    if current_pose is None or reference_pose is None:
        return False
    match_count = 0
    total_count = 0
    for i in range(len(reference_pose)):
        ref = reference_pose[i]
        cur = current_pose[i]
        if ref[0] is not None and cur[0] is not None:
            dist = np.linalg.norm(np.array(ref) - np.array(cur))
            total_count += 1
            if dist <= threshold_px:
                match_count += 1
    if total_count == 0:
        return False
    return (match_count / total_count) >= ratio

# YOLO 모델 로딩
yolo_model = YOLO("yolo11x-pose.pt")

# 기준 마스크 이미지 및 keypoint 로딩
## Trouble shooting : RGBA 형식이어야 해서 png로 꿀뷰를 사용해 변환함 -> 해결 안됨
### 해당 문제 해결을 위해, SAM 모델을 적용하면서 아예 RGBA(png의 형식)로 바뀌도록 함.
mask_image_path = 'C:/Users/user/Desktop/img_output/squat/mask/side_frame_000_mask.png'
mask = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)
print("mask shape:", mask.shape)
if mask is None:
    print("마스크 이미지를 찾을 수 없습니다.")
    exit()
reference_pose = load_reference_pose("C:/WANTED/LLM/KOMI_PJT/LLM_Project/LJH/data/side_json/frame_000.json")

# 마스크 오버레이 함수 (반투명 적용)
def overlay_mask(frame, mask, alpha_value=100):
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    
    # 마스크 반전 적용(왼쪽 위주로 찍히게끔)
    # mask_resized = cv2.flip(mask_resized, 1)
    mask_rgb = mask_resized[:, :, :3].astype(np.uint8)
    mask_alpha = mask_resized[:, :, 3].astype(np.uint8)
    object_mask = (mask_alpha > 0).astype(np.uint8)
    custom_alpha = np.full_like(mask_alpha, alpha_value, dtype=np.uint8)
    custom_alpha[object_mask == 0] = 0
    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    for c in range(3):
        frame_rgba[:, :, c] = (
            frame_rgba[:, :, c] * (1 - custom_alpha / 255.0) +
            mask_rgb[:, :, c] * (custom_alpha / 255.0)
        ).astype(np.uint8)
    frame_rgba[:, :, 3] = np.maximum(frame_rgba[:, :, 3], custom_alpha)
    return cv2.cvtColor(frame_rgba, cv2.COLOR_BGRA2BGR)

# 웹캠 설정 및 비디오 객체 생성
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_path, fourcc, 30.0, (1280, 720))
print("요청한 FPS: 30, 실제 적용된 FPS:", cap.get(cv2.CAP_PROP_FPS))

# 정렬 대기 루프
print("초기 정렬 대기 중...")
aligned = False
while not aligned:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    results = yolo_model.predict(source=frame, stream=False, verbose=False)
    keypoints = None
    for result in results:
        if result.keypoints is not None:
            keypoints_np = result.keypoints.xy.cpu().numpy()
            keypoints = keypoints_np[0]
            break
    vis_frame = frame.copy()
    vis_frame = overlay_mask(vis_frame, mask)
    if keypoints is not None:
        keypoints_array = np.array(keypoints, dtype=np.float32)
        if is_pose_similar_by_accuracy(keypoints_array, reference_pose):
            cv2.putText(vis_frame, "Pose Aligned! Starting soon...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            aligned = True
            time.sleep(1)
        else:
            cv2.putText(vis_frame, "Align your pose with the reference", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        cv2.putText(vis_frame, "Detecting pose...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    cv2.imshow("Pose Alignment", vis_frame)
    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()

# 카운트다운
print("정렬 완료! 5초 후 녹화가 시작됩니다.")
countdown_start = time.time()
while time.time() - countdown_start < 5:
    remaining = 5 - int(time.time() - countdown_start)
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    cv2.putText(frame, str(remaining), (600, 380), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
    cv2.imshow('Recording Video', frame)
    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()

# 2.9초간 녹화
record_start_time = time.time()
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    if time.time() - record_start_time <= 2.9:
        out.write(frame)
    else:
        print("2.9초 녹화 완료. 영상 저장 종료.")
        break
    cv2.imshow('Recording Video', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"녹화 완료: {video_path}")


# 87프레임 추출
cap = cv2.VideoCapture(video_path)
saved_idx = 0
frame_count = 0
while cap.isOpened() and frame_count < 87:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    image_name = f"frame_{saved_idx:03d}.jpg"
    output_image_path = os.path.join(image_output_folder, image_name)
    cv2.imwrite(output_image_path, frame)
    print(f"프레임 저장: {image_name}")
    saved_idx += 1
    frame_count += 1
cap.release()
cv2.destroyAllWindows()
print("모든 프레임 추출 및 저장 완료.")

# YOLO-Pose 적용 및 JSON 저장
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
image_files = sorted([f for f in os.listdir(image_output_folder) if f.endswith('.jpg')])
for image_file in image_files:
    image_path = os.path.join(image_output_folder, image_file)
    image = cv2.imread(image_path)
    results = yolo_model(image)
    json_data = {'image_name': image_file, 'bboxes': [], 'keypoints': []}
    keypoints_dict = {part: {"x": None, "y": None, "confidence": 0.0} for part in COCO_KEYPOINTS}
    for result in results:
        if result.boxes is not None:
            bboxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for bbox, conf, cls in zip(bboxes, confs, classes):
                x1, y1, x2, y2 = map(int, bbox)
                json_data['bboxes'].append({
                    'class': int(cls),
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf)
                })
        keypoints = result.keypoints.xy.cpu().numpy()
        scores = result.keypoints.conf.cpu().numpy()
        for idx, (kp, score) in enumerate(zip(keypoints[0], scores[0])):
            x, y = int(kp[0]), int(kp[1])
            conf = float(score)
            keypoints_dict[COCO_KEYPOINTS[idx]] = {
                'x': x if conf > 0.1 else None,
                'y': y if conf > 0.1 else None,
                "confidence": conf
            }
    json_data["keypoints"] = [
        {
            "part": part,
            "x": keypoints_dict[part]["x"],
            "y": keypoints_dict[part]["y"],
            "confidence": keypoints_dict[part]["confidence"]
        } for part in COCO_KEYPOINTS
    ]
    json_output_path = os.path.join(json_output_folder, image_file.replace('.jpg', '.json'))
    with open(json_output_path, 'w', encoding='utf-8') as jf:
        json.dump(json_data, jf, indent=4)
    print(f"YOLO-Pose 완료 및 JSON 저장: {json_output_path}")

# FFmpeg로 H.264 인코딩된 MP4로 변환
print("FFmpeg로 MP4(H.264)로 변환 중...")
ffmpeg_command = f'ffmpeg -y -i "{video_path}" -vcodec libx264 -crf 23 "{mp4_output}"'
os.system(ffmpeg_command)

# 변환 성공 여부 확인
if os.path.exists(mp4_output) and os.path.getsize(mp4_output) > 1000:
    print("MP4(H.264) 영상 저장 완료:", mp4_output)

    try :
        os.remove(f'{video_path}')
        print("임시 파일 삭제 완료")

    except:
        print("Error : Temp_file is not removed")
else:
    print("MP4 변환 실패 - FFmpeg 설치 또는 입력 확인 필요")
