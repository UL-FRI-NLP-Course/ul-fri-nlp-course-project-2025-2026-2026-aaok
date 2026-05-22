import time
import json
import os
import pandas as pd
import requests

INPUT_FILE = "data/filtered_pisrs.csv"
OUTPUT_FILE = "data/pisrs_metadata.jsonl"
API_URL = "https://pisrs.si/api/rezultat/zbirka/id/{}"

SLEEP_TIME = 0.1

df = pd.read_csv(INPUT_FILE)
moped_ids = df["mopedId"].dropna().unique().tolist()

done_ids = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                done_ids.add(json.loads(line)["mopedId"])
            except:
                pass

def fetch(mid):
    urls = [
        f"https://pisrs.si/api/rezultat/zbirka/id/{mid}",
        f"https://pisrs.si/api/rezultat/zbirka/neuradno-precisceno-besedilo/{mid}/{mid}",
    ]

    for url in urls:
        try:
            r = requests.get(url, timeout=15)

            if r.status_code == 200:
                return r.json()

        except:
            continue

    return None


start_time = time.time()
processed_times = []

total = len(moped_ids)
done_count = len(done_ids)

try:
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:

        for i, mid in enumerate(moped_ids):

            if mid in done_ids:
                continue

            t0 = time.time()
            data = fetch(mid)
            t1 = time.time()

            dt = t1 - t0
            processed_times.append(dt)

            if len(processed_times) > 50:
                processed_times.pop(0)

            avg_time = sum(processed_times) / len(processed_times)

            if data is not None:
                f.write(json.dumps({
                    "mopedId": mid,
                    "data": data
                }, ensure_ascii=False) + "\n")
                f.flush()

            done_count += 1

            elapsed = time.time() - start_time
            remaining = total - done_count
            eta = remaining * avg_time

            print(
                f"[{done_count}/{total}] "
                f"{mid} | "
                f"{dt:.2f}s | "
                f"avg {avg_time:.2f}s | "
                f"ETA {eta/60:.1f} min"
            )

            time.sleep(SLEEP_TIME)

except KeyboardInterrupt:
    print("\n\nInterrupted by user (Ctrl+C). Saving progress and exiting cleanly...")
    exit()
except Exception as e:
    print(e)
    exit()

print("Done scraping PISRS")
