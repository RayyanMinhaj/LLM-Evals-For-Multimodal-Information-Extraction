import json
from pathlib import Path
from openai import OpenAI

RESULTS_FILE = "llm_text_results.json"
TEXT_MULTI_PATH = "../m2e2(2)/text_multi.json"

SYSTEM_PROMPT = (
    "You are a strict trigger-word extractor. "
    "Given a sentence and an event type, output exactly the single word "
    "from the sentence that is the trigger for that event and nothing else."
)

client = OpenAI(api_key="")

with open(TEXT_MULTI_PATH) as f:
    data = json.load(f)

results_path = Path(RESULTS_FILE)
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)
    processed_ids = {r["sentence_id"] for r in results}
else:
    results = []
    processed_ids = set()

total = len(data)
for i, item in enumerate(data):
    sid = item["sentence_id"]
    if sid in processed_ids:
        continue

    sentence = item["sentence"]
    gems = item.get("golden-event-mentions", [])
    if not gems:
        print(f"[SKIP] no event mentions: {sid}")
        continue

    gem = gems[0]
    event_type = gem["event_type"]
    actual = gem["trigger"]["text"]

    user_prompt = f"Sentence: {sentence}\nEvent type: {event_type}\n\nWhat is the trigger word?"

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=16,
            temperature=0.0,
        )
        prediction = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] {sid}: {e}")
        continue

    results.append({
        "sentence_id": sid,
        "sentence": sentence,
        "predicted": prediction,
        "actual": actual,
    })
    print(f"[{i+1}/{total}] {sid}  ->  predicted: {prediction}  |  actual: {actual}")

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Results saved to {RESULTS_FILE} ({len(results)} total entries).")
