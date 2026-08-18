import os
import cv2
import numpy as np
import pandas as pd
import librosa
import mediapipe as mp

# Safe MoviePy import across versions
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Enables iris tracking landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Key Landmark Indices for Eye and Iris tracking
LEFT_IRIS = [469, 470, 471, 472]
LEFT_EYE_CORNER_OUTER = 33
LEFT_EYE_CORNER_INNER = 133

def extract_visual_features(video_path):
    """Tracks iris movement frame-by-frame and calculates horizontal gaze variance."""
    cap = cv2.VideoCapture(video_path)
    horizontal_gaze_ratios = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Left iris center coordinate
            iris_x = np.mean([landmarks[i].x for i in LEFT_IRIS])
            corner_outer_x = landmarks[LEFT_EYE_CORNER_OUTER].x
            corner_inner_x = landmarks[LEFT_EYE_CORNER_INNER].x
            
            # Normalize horizontal eye position relative to eye width
            eye_width = abs(corner_inner_x - corner_outer_x)
            if eye_width > 0:
                ratio = (iris_x - min(corner_outer_x, corner_inner_x)) / eye_width
                horizontal_gaze_ratios.append(ratio)

    cap.release()
    
    if len(horizontal_gaze_ratios) < 2:
        return 0.0, 0.0, 0.0

    ratios = np.array(horizontal_gaze_ratios)
    gaze_std = float(np.std(ratios))                           # Sweep variance
    gaze_diff = np.diff(ratios)
    saccade_velocity_mean = float(np.mean(np.abs(gaze_diff))) # Saccade speed
    saccade_velocity_std = float(np.std(gaze_diff))

    return gaze_std, saccade_velocity_mean, saccade_velocity_std

def extract_audio_features(video_path, temp_audio_path="temp_audio.wav"):
    """Extracts pause structure, speech pauses per min, and pitch jitter."""
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            video.close()
            return 0.0, 0.0, 0.0, 0.0
        
        video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
        video.close()

        y, sr = librosa.load(temp_audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        if duration <= 0:
            return 0.0, 0.0, 0.0, 0.0

        # Pause / Silence analysis (threshold: 25dB below peak)
        intervals = librosa.effects.split(y, top_db=25)
        speaking_time = sum([(end - start) / sr for start, end in intervals])
        silence_time = max(0.0, duration - speaking_time)
        pause_ratio = silence_time / duration
        pauses_per_min = len(intervals) / (duration / 60.0)

        # Pitch (F0) analysis using probabilistic YIN
        f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else []
        
        pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        pitch_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0

        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        return float(pause_ratio), float(pauses_per_min), pitch_mean, pitch_std
    except Exception as e:
        print(f"Audio processing error on {video_path}: {e}")
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return 0.0, 0.0, 0.0, 0.0

def process_dataset(video_folder="dataset/raw_videos", output_csv="data_features.csv"):
    if not os.path.exists(video_folder):
        print(f"Folder not found: {video_folder}")
        return

    files = [f for f in os.listdir(video_folder) if f.endswith((".mp4", ".mov", ".avi"))]
    if not files:
        print(f"No video files found in {video_folder}. Record or generate test videos first.")
        return

    data = []
    for filename in files:
        video_path = os.path.join(video_folder, filename)
        label = 1 if "scripted" in filename.lower() else 0  # 1 = Scripted, 0 = Spontaneous

        print(f"Processing: {filename} ...")
        g_std, s_vel_mean, s_vel_std = extract_visual_features(video_path)
        p_ratio, pauses_per_min, f0_mean, f0_std = extract_audio_features(video_path)

        data.append({
            "filename": filename,
            "label": label,
            "gaze_horizontal_std": round(g_std, 4),
            "saccade_velocity_mean": round(s_vel_mean, 4),
            "saccade_velocity_std": round(s_vel_std, 4),
            "pause_time_ratio": round(p_ratio, 4),
            "pauses_per_minute": round(pauses_per_min, 2),
            "pitch_mean_hz": round(f0_mean, 2),
            "pitch_std_hz": round(f0_std, 2)
        })

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"\nExtraction complete! Feature table saved to {output_csv}")

if __name__ == "__main__":
    process_dataset()