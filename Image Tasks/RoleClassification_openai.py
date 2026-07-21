import json
import os
import base64
from pathlib import Path
from openai import OpenAI

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MAX_IMAGES = 1318                 # how many images to process (increase when ready)
RESULTS_FILE = "llm_role_results.json"
IMAGE_DIR = "../cropped_images"
# ──────────────────────────────────────────────────────────────────────────────

ROLE_CLASSES = [
    "Agent",
    "Artifact",
    "Vehicle",
    "Destination",
    "Origin",
    "Attacker",
    "Target",
    "Instrument",
    "Place",
    "Entity",
    "Police",
    "Person",
    "Participant",
    "Victim",
    "Giver",
    "Recipient",
    "Money",
]

SYSTEM_PROMPT = (
    "You are a strict image classifier. "
    "Given an image, output exactly one of the following class labels and nothing else:\n"
    + "\n".join(ROLE_CLASSES)
)

client = OpenAI(api_key="")

image_paths = sorted(Path(IMAGE_DIR).iterdir())

# Load existing results so we can append
results_path = Path(RESULTS_FILE)
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)
    processed_ids = {r["image_id"] for r in results}
else:
    results = []
    processed_ids = set()

count = 0
for img_path in image_paths:
    if count >= MAX_IMAGES:
        break

    stem = img_path.stem
    if stem in processed_ids:
        continue

    if not img_path.exists():
        print(f"[SKIP] image not found: {img_path}")
        continue

    # Extract the class name from the filename: ..._{class_name}_{crop_idx}.jpg
    parts = stem.split("_")
    actual = parts[-2]

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "low",
                            },
                        },
                    ],
                },
            ],
            max_tokens=16,
            temperature=0.0,
        )
        prediction = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] {stem}: {e}")
        continue

    results.append({
        "image_id": stem,
        "predicted": prediction,
        "actual": actual,
    })
    print(f"[{count+1}/{MAX_IMAGES}] {stem}  \u2192  predicted: {prediction}  |  actual: {actual}")
    count += 1

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Results saved to {RESULTS_FILE} ({len(results)} total entries).")
