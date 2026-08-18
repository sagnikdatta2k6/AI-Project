import pandas as pd
import numpy as np

def generate_synthetic_features(n_samples=40, output_csv="data_features.csv"):
    np.random.seed(42)
    data = []

    for i in range(n_samples):
        # 50% scripted (1), 50% spontaneous (0)
        label = 1 if i % 2 == 0 else 0
        prefix = "scripted" if label == 1 else "spontaneous"
        filename = f"{prefix}_sample_{i+1}.mp4"

        if label == 1:
            # Scripted signatures: regular horizontal eye sweeps, fewer hesitations, flatter pitch
            gaze_std = np.random.normal(0.08, 0.015)
            saccade_vel = np.random.normal(0.025, 0.005)
            saccade_std = np.random.normal(0.015, 0.003)
            pause_ratio = np.random.normal(0.10, 0.03)
            pauses_per_min = np.random.normal(6.0, 1.5)
            pitch_mean = np.random.normal(165.0, 15.0)
            pitch_std = np.random.normal(18.0, 4.0)
        else:
            # Spontaneous signatures: stable gaze at camera/away to think, frequent pauses, varied pitch
            gaze_std = np.random.normal(0.03, 0.010)
            saccade_vel = np.random.normal(0.010, 0.003)
            saccade_std = np.random.normal(0.008, 0.002)
            pause_ratio = np.random.normal(0.28, 0.05)
            pauses_per_min = np.random.normal(14.0, 2.5)
            pitch_mean = np.random.normal(170.0, 15.0)
            pitch_std = np.random.normal(36.0, 6.0)

        data.append({
            "filename": filename,
            "label": label,
            "gaze_horizontal_std": max(0.0, round(float(gaze_std), 4)),
            "saccade_velocity_mean": max(0.0, round(float(saccade_vel), 4)),
            "saccade_velocity_std": max(0.0, round(float(saccade_std), 4)),
            "pause_time_ratio": max(0.0, round(float(pause_ratio), 4)),
            "pauses_per_minute": max(0.0, round(float(pauses_per_min), 2)),
            "pitch_mean_hz": round(float(pitch_mean), 2),
            "pitch_std_hz": max(0.0, round(float(pitch_std), 2))
        })

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Generated {n_samples} synthetic feature samples in {output_csv}")

if __name__ == "__main__":
    generate_synthetic_features()