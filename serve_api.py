# serve_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from pathlib import Path
import json
import re  # <-- added for text cleanup

MODEL_DIR = "model_output"
DATASET_FILENAME = "recipes_dataset.jsonl"

class Query(BaseModel):
    ingredients: str
    max_length: int = 150
    num_return_sequences: int = 1

app = FastAPI(title="Local Recipe Chatbot API")

# Global variables
generator = None
tokenizer = None
model = None
_recipe_map = {}  # maps normalized ingredient tuple -> recipe text

def _norm_ingredients(text: str):
    """Normalize ingredient string to a sorted tuple of lowercase tokens."""
    parts = [p.strip().lower() for p in text.split(",") if p.strip()]
    # remove duplicates and sort for stable key
    parts = sorted(set(parts))
    return tuple(parts)

# ✅ enhanced helper to clean recipe and keep title (used only for model-generated text)
def _format_steps(text: str, ingredients_hint: str = ""):
    """
    Convert raw model output into a formatted recipe:
    - Extracts a title (first line or phrase)
    - Converts messy text into numbered steps (1., 2., 3.)
    - Always shows a title (fallback uses ingredient names)
    """
    # remove redundant labels
    text = re.sub(r'(?i)\b(recipe|ingredients)[:\-]*', '', text).strip()

    # detect title (first line or sentence before a dot/newline)
    title_match = re.match(r'^([A-Z][^\n\.]{3,60})[\.:\n]', text)
    if title_match:
        title = title_match.group(1).strip()
        remaining = text[len(title):].strip()
    else:
        # fallback: build a readable title from ingredients_hint
        if ingredients_hint:
            items = [w.strip().capitalize() for w in re.split(r'[, ]+', ingredients_hint) if w.strip()]
            if len(items) > 1:
                title = " & ".join(items[:3]) + " Recipe"
            else:
                title = f"{items[0]} Recipe" if items else "Recipe"
        else:
            title = "Recipe"
        remaining = text

    # find steps (prefer numbered)
    lines = re.findall(r'\d+\.\s*[^0-9]+', remaining)
    if lines:
        steps = [ln.strip() for ln in lines]
    else:
        parts = re.split(r'[\.!?]', remaining)
        steps = [p.strip() for p in parts if len(p.strip()) > 5]

    # clean and limit steps
    steps = [f"{i+1}. {s}" for i, s in enumerate(steps[:8])]
    return f"🍽️ {title}\n\n" + "\n".join(steps).strip()

@app.on_event("startup")
def load_model_and_dataset():
    """Load dataset (into a fast lookup) and initialize model + pipeline."""
    global generator, tokenizer, model, _recipe_map
    base_dir = Path(__file__).resolve().parent

    # Load dataset and build quick lookup
    dataset_path = base_dir / DATASET_FILENAME
    if dataset_path.exists():
        try:
            with dataset_path.open("r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            print(f"✅ Found dataset: {dataset_path} with {len(lines)} entries")

            # Populate _recipe_map with normalized ingredient keys
            for entry in lines:
                text = entry.get("text", "")
                # Expecting "Ingredients: ...\nRecipe: ...", parse safely
                try:
                    parts = text.split("\n", 1)
                    first = parts[0] if len(parts) > 0 else ""
                    second = parts[1] if len(parts) > 1 else ""
                    # first should contain Ingredients: ...
                    if "ingredients:" in first.lower():
                        ing_part = first.split(":", 1)[1].strip()
                    else:
                        # fallback: try to extract from beginning
                        ing_part = first.strip()
                    # recipe is the remainder of the text (everything after Ingredients line)
                    recipe_text = text.replace(first + "\n", "", 1).strip()
                    key = _norm_ingredients(ing_part)
                    if key:
                        _recipe_map[key] = recipe_text
                except Exception:
                    # ignore malformed line
                    continue

            print(f"✅ Built quick-lookup map with {len(_recipe_map)} items")

        except Exception as e:
            print(f"⚠️ Failed to read dataset at {dataset_path}: {e}")
    else:
        print(f"⚠️ Dataset file not found at {dataset_path}. Create {DATASET_FILENAME} in project root for exact-match quick replies.")

    # Load fine-tuned model if present, otherwise fall back to distilgpt2 for speed
    model_dir_path = base_dir / MODEL_DIR
    try:
        if model_dir_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir_path))
            model = AutoModelForCausalLM.from_pretrained(str(model_dir_path))
            print(f"✅ Loaded fine-tuned model from {model_dir_path}")
        else:
            raise FileNotFoundError("model_output not found")
    except Exception as e:
        print(f"⚠️ Fine-tuned model not found or failed to load ({e}). Falling back to distilgpt2 (faster on CPU).")
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    # ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device selection: GPU (0) or CPU (-1)
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    print(f"✅ Generator pipeline ready (device={'cuda' if device==0 else 'cpu'})")

@app.post("/generate")
def generate(q: Query):
    """
    Generate a recipe for the given comma-separated ingredients.
    Implementation:
      1) Try exact/contained-match lookup in the small dataset -> immediate return (no model call).
      2) Otherwise, run a fast deterministic generation (greedy, small max_length) as fallback.
    """
    # Normalize incoming ingredients
    user_key = _norm_ingredients(q.ingredients)
    # 1) Exact / contained match: if any dataset key is subset of user ingredients OR equals, prefer closest match
    if user_key:
        # prefer exact match with same length
        if user_key in _recipe_map:
            # RETURN RAW DATASET RECIPE EXACTLY (do not reformat) so title+steps remain as in dataset
            return {"ingredients": q.ingredients, "recipes": [ _recipe_map[user_key] ]}

        # otherwise find dataset entries that are subset of user_key (i.e., dataset ingredients are contained in user's list)
        # prefer the longest subset match (most specific)
        best_key = None
        best_len = 0
        user_set = set(user_key)
        for k in _recipe_map.keys():
            kset = set(k)
            if kset.issubset(user_set):
                if len(k) > best_len:
                    best_len = len(k)
                    best_key = k
        if best_key:
            # RETURN RAW DATASET RECIPE EXACTLY (do not reformat)
            return {"ingredients": q.ingredients, "recipes": [ _recipe_map[best_key] ]}

    # 2) Fallback to model generation (fast configuration)
    try:
        # clamp max_length to a safe ceiling for CPU (60 tokens)
        max_len = min(q.max_length, 60)
        # stricter decoding to prevent repetition
        result = generator(
            f"Ingredients: {q.ingredients}\nRecipe:",
            max_length=max_len,
            num_return_sequences=1,
            do_sample=True,
            top_k=40,
            top_p=0.85,
            temperature=0.7,
            repetition_penalty=1.9,       # <-- added to stop loops
            pad_token_id=tokenizer.eos_token_id
        )

        prompt = f"Ingredients: {q.ingredients}\nRecipe:"
        recipes = []
        for r in result:
            txt = r.get("generated_text", "")
            if txt.startswith(prompt):
                txt = txt[len(prompt):].strip()
            txt = _format_steps(txt, q.ingredients)  # format only model-generated output
            recipes.append(txt)

        return {"ingredients": q.ingredients, "recipes": recipes}

    except Exception as e:
        # return a helpful error message (client will see this quickly)
        return {"error": f"generation failed: {e}"}

@app.get("/")
def home():
    return {"message": "API running. Use POST /generate or visit /docs for interactive docs."}
