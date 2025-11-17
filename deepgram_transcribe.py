import os
import json
from deepgram import DeepgramClient

# In PowerShell before running:
#    $env:DEEPGRAM_API_KEY = "040621178e7d629156aca413abe8434e74d2111d"

# ---------------------------
# CONFIGURATION
# ---------------------------
AUDIO_FILE = "bwc_4.mp3"
MODEL_NAME = "whisper-base"             # <- Change model here
CONFIDENCE_THRESHOLD = 0.90       # <- Change confidence filter here
OUTPUT_DIR = f"outputs/{MODEL_NAME}"


# ---------------------------
# HELPERS
# ---------------------------

def ensure_dir(path):
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)


def save_full_transcript(utterances, output_path):
    """Save standard diarized transcript."""
    with open(output_path, "w", encoding="utf-8") as f:
        for utt in utterances:
            start = utt.start
            end = utt.end
            speaker = utt.speaker or "Unknown"
            text = utt.transcript.strip()
            f.write(f"[{start:.2f} - {end:.2f}] Speaker {speaker}: {text}\n")


def save_confidence_filtered(utterances, output_path, threshold):
    """Save transcript where low-confidence words are replaced with '-'."""
    with open(output_path, "w", encoding="utf-8") as f:
        for utt in utterances:
            start = utt.start
            end = utt.end
            speaker = utt.speaker or "Unknown"

            filtered_words = []

            if hasattr(utt, "words") and utt.words:
                for w in utt.words:
                    if w.confidence and w.confidence >= threshold:
                        filtered_words.append(w.word)
                    else:
                        filtered_words.append("-")
            else:
                filtered_words.append(utt.transcript.strip())

            f.write(
                f"[{start:.2f} - {end:.2f}] Speaker {speaker}: {' '.join(filtered_words)}\n"
            )


# ---------------------------
# MAIN
# ---------------------------
def main():

    try:
        deepgram = DeepgramClient()
        ensure_dir(OUTPUT_DIR)

        with open(AUDIO_FILE, "rb") as audio:
            response = deepgram.listen.v1.media.transcribe_file(
                request=audio.read(),
                model=MODEL_NAME,
                smart_format=True,
                utterances=True,
                diarize=True,
                redact=["pii", "pci"],
                paragraphs=True
            )

        utterances = getattr(response.results, "utterances", None)

        if not utterances:
            print("❌ No utterances found.")
            return

        # ---------------------------
        # Save transcripts
        # ---------------------------

        full_path = os.path.join(OUTPUT_DIR, f"transcript_{MODEL_NAME}_full.txt")
        conf_path = os.path.join(
            OUTPUT_DIR,
            f"transcript_{MODEL_NAME}_conf{int(CONFIDENCE_THRESHOLD*100)}.txt"
        )

        save_full_transcript(utterances, full_path)
        save_confidence_filtered(utterances, conf_path, CONFIDENCE_THRESHOLD)

        print(f"\n✅ Saved full transcript: {full_path}")
        print(f"✅ Saved confidence-filtered transcript: {conf_path}\n")

    except Exception as e:
        print(f"❌ Exception: {e}")


if __name__ == "__main__":
    main()