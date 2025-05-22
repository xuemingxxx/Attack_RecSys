import pandas as pd
import random
import json
from tqdm import tqdm                                    # progress-bar

# ── Configuration ──────────────────────────────────────
datasets = [
    {   # Goodreads
        "name": "gr",
        "file": "raw_data/gr_book.csv",
        "target": 750_000
    },
    {   # 1-Million-Books
        "name": "1m",
        "file": "raw_data/1m_book.csv",
        "target": 750_000
    }
]
k = 4   # rating ≥ k is considered “liked”

# Load task templates (JSON list for each task)
with open("py_script/taskBookTemplate.txt", "r", encoding="utf-8") as f:
    tasks = json.load(f)

genders = ["male", "female"]
age_min, age_max = 18, 70
interest_tags_pool = [
    "Fantasy", "Mystery", "Romance", "Science Fiction", "Historical",
    "Thriller", "Biography", "Self-Help", "Classics", "Young Adult"
]

# ── Main generation loop ───────────────────────────────
combined_results = []

for ds in datasets:
    # Read the single CSV that already contains book & rating info
    df = pd.read_csv(ds["file"])
    user_ids = sorted(df["user_key"].unique())

    results = []
    idx = 0
    pbar = tqdm(total=ds["target"], desc=f"Generating {ds['name']} books")

    while len(results) < ds["target"]:
        uid = user_ids[idx % len(user_ids)]
        idx += 1
        user_data = df[df["user_key"] == uid]          # no timestamp sort needed

        # Split by preference
        high      = user_data[user_data["rating"] >= k]
        low       = user_data[user_data["rating"] <  k]
        high_list = list(high[["title", "book_key"]].itertuples(index=False, name=None))
        low_list  = low["title"].tolist()
        history   = list(user_data[["title", "rating"]].itertuples(index=False, name=None))

        # Random profile
        age    = random.randint(age_min, age_max)
        gender = random.choice(genders)

        # Random task / template
        task     = random.choice(list(tasks.keys()))
        template = random.choice(tasks[task])

        # Random n ∈ [3, 11]   (number of titles inside prompt)
        n = random.randint(3, 11)

        # ── Fill the template dynamically ─────────────────
        if task == "binary_classification":
            # 1 target + (n-1) titles; pref ≥ unpref, diff ≤ 1
            unpref_need = (n - 1) // 2
            pref_need   = (n - 1) - unpref_need
            if len(high_list) < pref_need or len(low_list) < unpref_need:
                continue

            pref_sample   = random.sample(high_list, pref_need)
            unpref_sample = random.sample(low_list,  unpref_need)

            book_names = ", ".join(f'"{t}"' for t, _ in pref_sample)
            book_ids   = ", ".join(str(bid) for _, bid in pref_sample)

            candidates = df[~df["book_key"].isin(user_data["book_key"])]
            if candidates.empty:
                candidates = df
            tgt = candidates.sample(1).iloc[0]

            prompt = template.format(
                age=age, gender=gender,
                pref=book_names,
                unpref=", ".join(f'"{t}"' for t in unpref_sample),
                target=f'"{tgt["title"]}"',
                user_id=uid, book_names=book_names,
                book_ids=book_ids,
                target_title=tgt["title"], target_id=tgt["book_key"]
            )

        elif task == "direct_recommendation":
            if len(high_list) < n:
                continue
            pref_sample = random.sample(high_list, n)
            prompt = template.format(
                age=age, gender=gender,
                pref=", ".join(f'"{t}"' for t, _ in pref_sample)
            )

        elif task == "sequential_recommendation":
            if len(history) < n:
                continue
            his = history[:n]
            prompt = template.format(
                age=age, gender=gender,
                his=", ".join(f'"{t}"' for t, _ in his)
            )

        elif task == "rating_prediction":
            his_need = n - 1                      # leave 1 slot for target
            if len(history) < his_need or his_need <= 0:
                continue
            his_r   = history[:his_need]
            rec_str = ", ".join(f'"{t}"-{r}' for t, r in his_r)

            candidates = df[~df["book_key"].isin(user_data["book_key"])]
            if candidates.empty:
                candidates = df
            tgt = candidates.sample(1).iloc[0]

            prompt = template.format(
                rating_rec=rec_str,
                his="; ".join(f'"{t}"' for t, _ in his_r),
                rating=his_r[0][1],
                targetTitle=f'"{tgt["title"]}"'
            )

        else:  # cold_start (no item titles, still use 3 interest tags)
            tags = random.sample(interest_tags_pool, 3)
            prompt = template.format(age=age, gender=gender, pref=", ".join(tags))

        # Save and update progress
        results.append({"text": prompt})
        pbar.update(1)

    pbar.close()
    combined_results.extend(results)

# ── Write merged output ────────────────────────────────
with open("invInst_book.json", "w", encoding="utf-8") as f:
    json.dump(combined_results, f, ensure_ascii=False, indent=2)
print(f"Finished: {len(combined_results)} prompts saved to invInst_book.json")
