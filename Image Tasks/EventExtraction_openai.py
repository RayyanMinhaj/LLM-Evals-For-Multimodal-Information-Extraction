import json
import os
import base64
from pathlib import Path
from openai import OpenAI

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MAX_IMAGES = 395                 # how many images to process (increase when ready)
RESULTS_FILE = "llm_results.json"
IMG_TRAIN_PATH = "../m2e2(2)/img_train.json"
IMAGE_DIR = "../m2e2(2)/image/image"
# ──────────────────────────────────────────────────────────────────────────────

EVENT_CLASSES = [
    "Movement:Transport",
    "Conflict:Attack",
    "Conflict:Demonstrate",
    "Justice:Arrest-Jail",
    "Contact:Phone-Write",
    "Contact:Meet",
    "Life:Die",
    "Transaction:Transfer-Money",
]

SYSTEM_PROMPT = (
    "You are a strict image classifier. "
    "Given an image, output exactly one of the following class labels and nothing else:\n"
    + "\n".join(EVENT_CLASSES)
)

client = OpenAI(api_key="")

with open(IMG_TRAIN_PATH) as f:
    img_data = json.load(f)

image_ids = list(img_data.keys())

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
for img_id in image_ids:
    if count >= MAX_IMAGES:
        break
    if img_id in processed_ids:
        continue

    img_path = Path(IMAGE_DIR) / f"{img_id}.jpg"
    if not img_path.exists():
        print(f"[SKIP] image not found: {img_path}")
        continue

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
        print(f"[ERROR] {img_id}: {e}")
        continue

    actual = img_data[img_id]["event_type"]

    results.append({
        "image_id": img_id,
        "predicted": prediction,
        "actual": actual,
    })
    print(f"[{count+1}/{MAX_IMAGES}] {img_id}  →  predicted: {prediction}  |  actual: {actual}")
    count += 1

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Results saved to {RESULTS_FILE} ({len(results)} total entries).")
