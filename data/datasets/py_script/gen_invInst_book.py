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
    books_df = pd.read_csv(ds["file"])
    user_ids = sorted(books_df["user_key"].unique())

    results = []
    idx = 0
    pbar = tqdm(total=ds["target"], desc=f"Generating {ds['name']} books")

    while len(results) < ds["target"]:
        uid = user_ids[idx % len(user_ids)]
        idx += 1
        user_data = books_df[books_df["user_key"] == uid]          # no timestamp sort needed

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
            need_pref        = "{pref}"        in template
            need_unpref      = "{unpref}"      in template
            need_book_names  = "{book_names}"  in template
            need_book_ids    = "{book_ids}"    in template
            need_uid         = "{user_id}"     in template

            usable_n = n - 1        # 1 个 target，剩下 usable_n 个历史图书

            if need_pref or need_unpref:                       # pref / unpref 版本
                unpref_need = usable_n // 2                    # floor
                pref_need   = usable_n - unpref_need           # ceil
            else:                                              # 只有 book_names
                pref_need   = usable_n
                unpref_need = 0

            if len(high_list) < pref_need or len(low_list) < unpref_need:
                continue

            pref_sample   = random.sample(high_list, pref_need)
            unpref_sample = random.sample(low_list,  unpref_need) if unpref_need else []
            pref_names   = ", ".join(f'"{t}"' for t, _ in pref_sample)
            pref_ids     = ", ".join(str(mid) for _, mid in pref_sample)
            unpref_names = ", ".join(f'"{t}"' for t in unpref_sample)

            book_names_str = pref_names         # 仅喜欢图书的集合
            book_ids_str   = pref_ids

            candidates = books_df[~books_df["book_key"].isin(user_data["book_key"])]
            if candidates.empty:
                candidates = books_df
            tgt = candidates.sample(1).iloc[0]

            fmt = {
                "age": age,
                "gender": gender,
                "pref": pref_names,
                "unpref": unpref_names,
                "book_names": book_names_str,
                "book_ids": book_ids_str if need_book_ids else "",
                "user_id": uid if need_uid else "",
                "target": f'"{tgt["title"]}"',
                "target_id": tgt["book_key"],
            }
            prompt = template.format(**fmt)

        elif task == "direct_recommendation":
            need_pref        = "{pref}"        in template
            need_unpref      = "{unpref}"      in template
            need_book_names  = "{book_names}"  in template
            need_book_ids    = "{book_ids}"    in template
            need_uid         = "{user_id}"     in template

            if need_pref or need_unpref:                      # pref / unpref 版本
                unpref_need = n // 2                          # floor(n/2)
                pref_need   = n - unpref_need                 # 剩下给 pref（奇数时多 1）
            else:                                             # book_names 版本
                pref_need   = n                               # 全部都是喜欢的图书
                unpref_need = 0

            if len(high_list) < pref_need or len(low_list) < unpref_need:
                continue

            pref_sample   = random.sample(high_list, pref_need)
            unpref_sample = random.sample(low_list,  unpref_need) if unpref_need else []
            pref_names   = ", ".join(f'"{t}"' for t, _ in pref_sample)
            pref_ids     = ", ".join(str(mid) for _, mid in pref_sample)
            unpref_names = ", ".join(f'"{t}"' for t in unpref_sample)

            book_names_str = pref_names
            book_ids_str   = pref_ids
            fmt = {
                "age": age,
                "gender": gender,
                "pref": pref_names,
                "unpref": unpref_names,
                "book_names": book_names_str,
                "book_ids": book_ids_str if need_book_ids else "",
                "user_id": uid if need_uid else "",
            }
            prompt = template.format(**fmt)

        elif task == "sequential_recommendation":
            if len(history) < n:
                continue
            his = history[:n]
            prompt = template.format(
                age=age, gender=gender,
                his=", ".join(f'"{t}"' for t, _ in his)
            )

        elif task == "rating_prediction":
            his_need = n - 1          # reserve 1 title for target
            if len(history) < his_need or his_need <= 0:
                continue
            his_r   = history[:his_need]
            rec_str = ", ".join(f'"{t}"-{r}' for t, r in his_r)

            candidates = books_df[~books_df["book_key"].isin(user_data["book_key"])]
            if candidates.empty:
                candidates = books_df
            tgt = candidates.sample(1).iloc[0]

            prompt = template.format(
                rating_rec=rec_str,
                his="; ".join(f'"{t}"' for t, _ in his_r),
                rating=his_r[0][1],
                targetTitle=f'"{tgt["title"]}"'
            )

        elif task == "cold_start":
            need_book_names = "{book_names}" in template
            need_taste      = "{taste}"       in template
            need_book_ids   = "{book_ids}"    in template
            need_uid        = "{user_id}"     in template   

            if need_book_names:
                if len(high_list) < n:
                    continue
                book_sample     = random.sample(high_list, n)
                book_names_str  = ", ".join(f'"{t}"' for t, _ in book_sample)
                book_ids_str    = ", ".join(str(mid) for _, mid in book_sample)
            else:
                book_names_str = ""
                book_ids_str   = ""

            if need_taste:
                tags = random.sample(interest_tags_pool, 3)
                taste_str = ", ".join(tags)
            else:
                taste_str = ""

            fmt = {
                "age": age,
                "gender": gender,
                "book_names": book_names_str,
                "book_ids": book_ids_str if need_book_ids else "",
                "taste": taste_str,
                "user_id": uid if need_uid else "",
            }
            prompt = template.format(**fmt)

        # Store result and update progress bar
        results.append({"text": prompt})
        pbar.update(1)
    pbar.close()
    combined_results.extend(results)

# ── Write merged output ────────────────────────────────
with open("invInst_book.json", "w", encoding="utf-8") as f:
    json.dump(combined_results, f, ensure_ascii=False, indent=2)
print(f"Finished: {len(combined_results)} prompts saved to invInst_book.json")
