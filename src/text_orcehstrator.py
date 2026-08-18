from orchestrator import MultiAgentFusionOrchestrator
import json

# Initialize the Multi-Agent Orchestrator
orchestrator = MultiAgentFusionOrchestrator()

# When you have an actual test video:
# report = orchestrator.process_video_interview("dataset/raw_videos/scripted_1.mp4")
# print(json.dumps(report, indent=2))

print("\nAll 3 sub-agents and the Master Orchestrator loaded successfully.")