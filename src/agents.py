import os
import cv2
import numpy as np
import librosa
import mediapipe as mp
import whisper
import spacy

nlp = spacy.load("en_core_web_sm")
whisper_model = whisper.load_model("tiny")  # Lightweight, fast ASR

# --- AGENT 1: VISION AGENT ---
class VisionAgent:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        )
        self.left_iris = [469, 470, 471, 472]
        self.left_corner_outer = 33
        self.left_corner_inner = 133

    def analyze(self, video_path):
        cap = cv2.VideoCapture(video_path)
        gaze_ratios = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                iris_x = np.mean([lm[i].x for i in self.left_iris])
                outer_x = lm[self.left_corner_outer].x
                inner_x = lm[self.left_corner_inner].x
                w = abs(inner_x - outer_x)
                if w > 0:
                    gaze_ratios.append((iris_x - min(outer_x, inner_x)) / w)
        cap.release()

        gaze_std = float(np.std(gaze_ratios)) if len(gaze_ratios) > 1 else 0.0
        # High horizontal gaze variance suggests eye sweep across text lines
        is_suspicious = gaze_std > 0.05
        return {
            "gaze_horizontal_variance": round(gaze_std, 4),
            "visual_verdict": "Suspicious (Reading Gaze Sweep)" if is_suspicious else "Normal (Natural Fixation)"
        }

# --- AGENT 2: AUDIO AGENT ---
class AudioAgent:
    def analyze(self, audio_path):
        if not os.path.exists(audio_path):
            return {"audio_verdict": "No audio file", "pause_ratio": 0.0, "pitch_variance": 0.0}
        
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration <= 0:
            return {"audio_verdict": "Empty audio", "pause_ratio": 0.0, "pitch_variance": 0.0}

        # Pauses detection
        intervals = librosa.effects.split(y, top_db=25)
        speaking_time = sum([(end - start) / sr for start, end in intervals])
        pause_ratio = max(0.0, (duration - speaking_time) / duration)

        # Pitch variation (monotone delivery vs natural inflection)
        f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitch_std = float(np.std(f0[voiced])) if (voiced is not None and np.sum(voiced) > 0) else 0.0

        # Scripted reading typically lacks thinking pauses and has flatter pitch
        is_suspicious = (pause_ratio < 0.15) and (pitch_std < 25.0)
        return {
            "pause_ratio": round(pause_ratio, 3),
            "pitch_std_hz": round(pitch_std, 2),
            "audio_verdict": "Suspicious (Continuous / Monotone Delivery)" if is_suspicious else "Normal (Natural Pausing & Pitch)"
        }

# --- AGENT 3: LINGUISTIC AGENT ---
class LinguisticAgent:
    def analyze(self, audio_path):
        if not os.path.exists(audio_path):
            return {"transcript": "", "filler_ratio": 0.0, "text_verdict": "No audio"}
        
        # Whisper Transcription
        result = whisper_model.transcribe(audio_path)
        transcript = result.get("text", "").strip()

        doc = nlp(transcript.lower())
        tokens = [token.text for token in doc if not token.is_punct]
        filler_words = {"um", "uh", "like", "you know", "actually", "basically", "so"}
        
        filler_count = sum(1 for token in tokens if token in filler_words)
        total_tokens = max(len(tokens), 1)
        filler_ratio = filler_count / total_tokens

        # Spontaneous speech naturally has hesitations / filler words; reading AI text rarely does
        is_suspicious = (filler_ratio < 0.02) and (total_tokens > 20)
        return {
            "transcript": transcript,
            "word_count": total_tokens,
            "filler_ratio": round(filler_ratio, 4),
            "text_verdict": "Suspicious (Unusually Fluent / Written Grammar)" if is_suspicious else "Normal (Spontaneous Hesitations Present)"
        }