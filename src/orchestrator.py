import os
import json
from moviepy import VideoFileClip
from agents import VisionAgent, AudioAgent, LinguisticAgent

class MultiAgentFusionOrchestrator:
    def __init__(self):
        print("Initializing AI Sub-Agents...")
        self.vision_agent = VisionAgent()
        self.audio_agent = AudioAgent()
        self.linguistic_agent = LinguisticAgent()

    def process_video_interview(self, video_path):
        print(f"\n--- Orchestrating Analysis for: {os.path.basename(video_path)} ---")
        
        # 1. Split video track into temporary audio for audio/text agents
        temp_audio = "temp_orchestrator_audio.wav"
        video = VideoFileClip(video_path)
        if video.audio:
            video.audio.write_audiofile(temp_audio, verbose=False, logger=None)
        video.close()

        # 2. Parallel / Sequential Sub-Agent Execution
        print("[1/3] Vision Agent evaluating eye saccades & gaze sweep...")
        vision_report = self.vision_agent.analyze(video_path)

        print("[2/3] Audio Agent evaluating prosody & pause ratios...")
        audio_report = self.audio_agent.analyze(temp_audio)

        print("[3/3] Linguistic Agent running Whisper transcription & disfluency analysis...")
        linguistic_report = self.linguistic_agent.analyze(temp_audio)

        if os.path.exists(temp_audio):
            os.remove(temp_audio)

        # 3. Decision Fusion (Weighted Multi-Agent Voting)
        suspicion_score = 0
        signals = []

        if "Suspicious" in vision_report["visual_verdict"]:
            suspicion_score += 40
            signals.append("Vision Agent detected repetitive horizontal gaze sweep consistent with reading off-screen.")
        
        if "Suspicious" in audio_report["audio_verdict"]:
            suspicion_score += 30
            signals.append("Audio Agent detected lack of natural thinking pauses and low pitch variance.")

        if "Suspicious" in linguistic_report["text_verdict"]:
            suspicion_score += 30
            signals.append("Linguistic Agent detected hyper-polished syntax with zero natural filler hesitations.")

        # 4. Final Aggregated Report
        final_classification = "Likely Scripted / AI-Read" if suspicion_score >= 50 else "Likely Spontaneous"

        final_report = {
            "file": os.path.basename(video_path),
            "final_verdict": final_classification,
            "confidence_score": f"{suspicion_score}%",
            "agent_findings": {
                "vision": vision_report,
                "audio": audio_report,
                "linguistics": linguistic_report
            },
            "orchestrator_synthesis": signals if signals else ["All modalities reflect natural, unscripted responses."]
        }

        return final_report

if __name__ == "__main__":
    orchestrator = MultiAgentFusionOrchestrator()
    print("\nMulti-Agent Orchestrator is ready to run on interview recordings.")