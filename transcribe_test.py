import os
import time
import whisper

# Add local ffmpeg folder to PATH
ffmpeg_path = os.path.join(os.getcwd(), "ffmpeg")
os.environ["PATH"] += os.pathsep + ffmpeg_path

# Load the model
model = whisper.load_model("base")

# Path to your video

video_path = "data/bwc_4.mp4"

# Start timer
start_time = time.time()

# Transcribe with timestamps
result = model.transcribe(video_path, task="transcribe", verbose=False)

# End timer
end_time = time.time()

# Print transcription with timestamps
print("\nTranscription with timestamps:")
for segment in result["segments"]:
    start = segment["start"]
    end = segment["end"]
    text = segment["text"].strip()
    print(f"[{start:.2f}s - {end:.2f}s]: {text}")

# Print total time taken
duration = end_time - start_time
print(f"\nTranscription took {duration:.2f} seconds")
