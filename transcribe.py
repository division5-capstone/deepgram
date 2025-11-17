 
import os
import json
from deepgram import DeepgramClient
import re
 
# 🧠 Make sure your API key is set:
# In PowerShell before running:
#    $env:DEEPGRAM_API_KEY = "your_api_key_here"
 
VIDEO_FILE = "bwc_4.mp3"
 
def main():
 
    try:
        # ✅ Correct initialization for Deepgram SDK 5.3.0
        deepgram = DeepgramClient()
 
        # Open your video file in binary mode
        with open(VIDEO_FILE, "rb") as audio_file:
 
            # Send request if local file
            response = deepgram.listen.v1.media.transcribe_file(                
                request=audio_file.read(),
                model="nova-2",
                smart_format=True,
                utterances= True,
                diarize=True
            )
 
        if hasattr(response.results, "utterances") and response.results.utterances:
            with open("transcript_nova_2.txt", "w", encoding="utf-8") as f:
                for utt in response.results.utterances:
                    start = utt.start
                    end = utt.end
                    speaker = utt.speaker if utt.speaker is not None else "Unknown"
                    text = utt.transcript.strip()
 
                    # Write in readable format
                    f.write(f"[{start:.2f} - {end:.2f}] Speaker {speaker}: {text}\n")
 
            print("\n✅ Timestamped transcript with speaker diarization saved as transcript_updated.txt")
        else:
            print("❌ No utterances found in the response.")
 
        # ✅ Extract utterances with speaker and timestamps
        if hasattr(response.results, "utterances") and response.results.utterances:
            with open("transcript_with_confidence_90.txt", "w", encoding="utf-8") as f:
                for utt in response.results.utterances:
                    start = utt.start
                    end = utt.end
                    speaker = utt.speaker if utt.speaker is not None else "Unknown"
 
                    # Build transcript with confidence filtering
                    filtered_words = []
                    if hasattr(utt, "words") and utt.words:
                        for word in utt.words:
                            if word.confidence is not None and word.confidence >= 0.9:
                                filtered_words.append(word.word)
                            else:
                                filtered_words.append("-")
                    else:
                        # fallback: if no word-level data, just use utterance text
                        filtered_words.append(utt.transcript.strip())
 
                    filtered_text = " ".join(filtered_words)
                    f.write(f"[{start:.2f} - {end:.2f}] Speaker {speaker}: {filtered_text}\n")
 
            print("\n✅ Transcript (filtered by 90% confidence) saved as transcript_updated.txt")
 
        else:
            print("❌ No utterances found in the response.")
 
    except Exception as e:
        print(f"❌ Exception: {e}")
 
if __name__ == "__main__":
    main()
 
 
 