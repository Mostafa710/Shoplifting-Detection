import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
from torchvision.models.video import r3d_18

# === Model + Preprocessing must match training notebook ===
NUM_FRAMES = 16
IMG_SIZE = 112
CLASS_NAMES = ["Normal", "Shoplifting"]  # 0, 1 per notebook

# Per-frame transforms (no normalization in the notebook)
frame_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def sample_uniform_frames(video_path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames >= num_frames:
        frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    else:
        frame_indices = list(range(total_frames)) + [max(total_frames - 1, 0)] * (num_frames - total_frames)

    frames = []
    last_frame = None
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # fallback to last seen frame or black
            frame = last_frame if last_frame is not None else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_frame = frame
        frames.append(frame)
    cap.release()
    return frames

def build_model(num_classes=2, dropout_p=0.5):
    model = r3d_18(pretrained=True)
    in_f = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_f, num_classes))
    return model

def load_finetuned_model(weights_path: str, device: str = "cpu"):
    model = build_model()
    state = torch.load(weights_path, map_location=torch.device(device))
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model

def predict_video(model, video_path: str, device: str = "cpu"):
    frames = sample_uniform_frames(video_path)
    frames_t = [frame_transform(f) for f in frames]  # list of (C,H,W)
    clip = torch.stack(frames_t).permute(1, 0, 2, 3).unsqueeze(0)  # (1,C,T,H,W)
    clip = clip.to(device)

    with torch.no_grad():
        logits = model(clip)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
        pred_idx = int(np.argmax(probs))
    return { "pred_idx": pred_idx, "label": CLASS_NAMES[pred_idx], "probs": probs.tolist() }
