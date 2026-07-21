import json
from pathlib import Path
import ollama

RESULTS_FILE = "llm_text_role_results_ollama.json"
TEXT_MULTI_PATH = "../m2e2(2)/text_multi.json"

ROLE_CLASSES = [
    "Agent", "Artifact", "Vehicle", "Destination", "Origin",
    "Attacker", "Target", "Instrument", "Place", "Entity",
    "Police", "Person", "Participant", "Victim", "Giver",
    "Recipient", "Money",
]

SYSTEM_PROMPT = (
    "You are a strict role classifier. "
    "Given a sentence, an event type, and a text span from the sentence, "
    "output exactly one of the following role labels and nothing else:\n"
    + "\n".join(ROLE_CLASSES)
)

MODEL = "gemma3:12b"

with open(TEXT_MULTI_PATH) as f:
    data = json.load(f)

results_path = Path(RESULTS_FILE)
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)
    processed_ids = {(r["sentence_id"], r["arg_text"]) for r in results}
else:
    results = []
    processed_ids = set()

total_args = sum(len(gem.get("arguments", [])) for item in data for gem in item.get("golden-event-mentions", []))
count = 0
for item in data:
    sid = item["sentence_id"]
    sentence = item["sentence"]
    for gem in item.get("golden-event-mentions", []):
        event_type = gem["event_type"]
        for arg in gem.get("arguments", []):
            arg_text = arg["text"]
            actual_role = arg["role"]

            if (sid, arg_text) in processed_ids:
                continue

            user_prompt = f"Sentence: {sentence}\nEvent type: {event_type}\nText span: \"{arg_text}\"\n\nWhat is the role of this text span?"

            try:
                response = ollama.chat(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    options={"temperature": 0.0},
                )
                prediction = response["message"]["content"].strip()
            except Exception as e:
                print(f"[ERROR] {sid} arg=\"{arg_text}\": {e}")
                continue

            results.append({
                "sentence_id": sid,
                "arg_text": arg_text,
                "predicted": prediction,
                "actual": actual_role,
            })
            count += 1
            print(f"[{count}/{total_args}] {sid} arg=\"{arg_text}\"  ->  predicted: {prediction}  |  actual: {actual_role}")

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Results saved to {RESULTS_FILE} ({len(results)} total entries).")
