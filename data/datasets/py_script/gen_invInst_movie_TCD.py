import json
import random
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────
MOVIES_FILE   = "raw_data/movies.json"   # single catalogue with {"movieId", "title"}
TOTAL_PROMPTS = 1_500_000                # overall number of prompts to generate

# Load task templates (JSON list for each task)
with open("py_script/taskMovieTemplate.txt", "r", encoding="utf-8") as f:
    tasks = json.load(f)

genders = ["male", "female"]
age_min, age_max = 18, 70
interest_tags_pool = [
    "Action", "Comedy", "Drama", "Sci-Fi", "Fantasy",
    "Romance", "Thriller", "Horror", "Documentary", "Animation"
]

# ── Prepare movie catalogue ────────────────────────────
with open(MOVIES_FILE, "r", encoding="utf-8") as f:
    movies_catalog = json.load(f)        # list of dicts
all_movies = [(m["title"], m["movieId"]) for m in movies_catalog]

# ── Main generation loop ───────────────────────────────
results = []
pbar = tqdm(total=TOTAL_PROMPTS, desc="Generating prompts")

while len(results) < TOTAL_PROMPTS:
    # Synthetic user profile
    age     = random.randint(age_min, age_max)
    gender  = random.choice(genders)
    user_id = random.randint(1, 10_000_000)

    # Random task/template
    task     = random.choice(list(tasks.keys()))
    template = random.choice(tasks[task])

    # n titles inside the prompt
    n = random.randint(3, 11)

    # Draw unique movie samples (extra one for target if needed)
    sample_pool = random.sample(all_movies, n + 1) if task in {"binary_classification", "rating_prediction"} else random.sample(all_movies, n)

    # ── Fill template ──────────────────────────────────
    if task == "binary_classification":
        target_title, target_id = sample_pool[0]
        remaining               = sample_pool[1:]

        unpref_need = (n - 1) // 2
        pref_need   = (n - 1) - unpref_need
        pref_sample = remaining[:pref_need]
        unpref_sample = remaining[pref_need:]

        movie_names = ", ".join(f'"{t}"' for t, _ in pref_sample)
        movie_ids   = ", ".join(str(mid) for _, mid in pref_sample)

        prompt = template.format(
            age=age, gender=gender,
            pref=movie_names,
            unpref=", ".join(f'"{t}"' for t, _ in unpref_sample),
            target=f'"{target_title}"',
            user_id=user_id, movie_names=movie_names,
            movie_ids=movie_ids,
            target_title=f'"{target_title}"', target_id=target_id
        )

    elif task == "direct_recommendation":
        pref_sample = sample_pool
        prompt = template.format(
            age=age, gender=gender,
            pref=", ".join(f'"{t}"' for t, _ in pref_sample)
        )

    elif task == "sequential_recommendation":
        his = sample_pool
        prompt = template.format(
            age=age, gender=gender,
            his=", ".join(f'"{t}"' for t, _ in his)
        )

    elif task == "rating_prediction":
        his_need = n - 1
        his_r    = sample_pool[:his_need]
        target_title, _ = sample_pool[-1]

        rec_str = ", ".join(f'"{t}"-{random.randint(1,5)}' for t, _ in his_r)
        prompt = template.format(
            rating_rec=rec_str,
            his="; ".join(f'"{t}"' for t, _ in his_r),
            rating=random.randint(1,5),
            targetTitle=f'"{target_title}"'
        )

    else:  # cold_start (no titles, use 3 interest tags)
        tags = random.sample(interest_tags_pool, 3)
        prompt = template.format(age=age, gender=gender, pref=", ".join(tags))

    results.append({"text": prompt})
    pbar.update(1)

pbar.close()

# ── Output ─────────────────────────────────────────────
with open("invInst_movie1.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Finished: {len(results)} prompts saved to invInst_movie.json")
