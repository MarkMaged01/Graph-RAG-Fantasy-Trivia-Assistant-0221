import os

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/dev/null"


from typing import Dict, Any, List
import time, json, re, sys, os
from huggingface_hub import login
from huggingface_hub import InferenceClient

# =============================
# MODEL CONFIG
# =============================
MODEL_REGISTRY = {
    "gemma": "google/gemma-2-2b-it",
    "zephyr": "HuggingFaceH4/zephyr-7b-beta",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2"
}

# =============================
# MODEL CAPABILITIES
# =============================
CHAT_MODELS = {
    "gemma",
    "zephyr",
    "mistral"  # All three are chat models
}

TEXT_MODELS = set()  # No text-only models

def get_generation_kwargs(model_key, models):
    tok = models.tokenizers[model_key]
    return {
        "max_new_tokens": 384,
        "do_sample": False,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.eos_token_id,
        "return_full_text": False,
    }

# =============================
# HF SETUP
# =============================
HF_TOKEN = os.getenv("HF_TOKEN")


class HFModels:
    def __init__(self, token=None):
        self.clients = {}
        self.token = token

    def load_models(self, only=None):
        for key, model_id in MODEL_REGISTRY.items():
            try:
                # FIXED: Removed provider="hf-inference" - this was causing the 404
                self.clients[key] = InferenceClient(
                    model=model_id,
                    token=self.token
                )
                print(f"✓ Loaded {key}: {model_id}")
            except Exception as e:
                print(f"✗ Failed to load {key}: {e}")

    def is_ready(self, key):
        return key in self.clients

    def generate(self, key, prompt, max_new_tokens=256):
        if key not in self.clients:
            raise ValueError(f"Model {key} not loaded")
        
        try:
            print(f"USING CHAT COMPLETION FOR {key}")
            response = self.clients[key].chat_completion(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_new_tokens,
                temperature=0.0
            )
            # Access content correctly (it's .content not ["content"])
            return response.choices[0].message.content
                
        except Exception as e:
            print(f"Error generating with {key}: {e}")
            import traceback
            traceback.print_exc()
            return f"[ERROR: {str(e)}]"

if HF_TOKEN.strip():
    login(token=HF_TOKEN)
    print("HF login successful")
else:
    print("⚠ No HF token provided. Public models only.")



# =============================
# UTIL
# =============================
def safe_str(x):
    try:
        return str(x)
    except:
        return json.dumps(x)

# =============================
# MODEL LOADER
# =============================
def validate_baseline(baseline: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanity-check baseline stats to avoid impossible values
    """
    if not baseline:
        return baseline

    cleaned = dict(baseline)

    # Goals sanity check
    if "Goals" in cleaned:
        try:
            goals = int(cleaned["Goals"])
            if goals > 70:  # clearly impossible for a season
                cleaned["_warning"] = " "
        except:
             cleaned["_warning"] = "Goals value non-numeric"

    return cleaned


# =============================
# CONTEXT MERGE
# =============================
def merge_context(baseline, embeddings, keep_top_k=5):
    lines = []
    sources = []

    if baseline:
        if isinstance(baseline, list):
            for i, row in enumerate(baseline):
                row = validate_baseline(row)
                lines.append(f"BASE[{i}] {safe_str(row)}")
                sources.append(("baseline", i))
        else:
            baseline = validate_baseline(baseline)
            lines.append(f"BASE[0] {safe_str(baseline)}")
            sources.append(("baseline", 0))

    if embeddings:
        for i, e in enumerate(embeddings[:keep_top_k]):
            lines.append(f"EMB[{i}] score={e.get('score')} | {e.get('text')}")
            sources.append(("embedding", i))

    return {
        "context_text": "\n".join(lines),
        "sources": sources
    }

# =============================
# PROMPT BUILDER
# =============================
SYSTEM_PROMPT = (
    "You are an FPL assistant. Answer ONLY using the CONTEXT. "
    "You may list or summarize available statistics from the context. "
    "If the required information is missing, say 'Not enough data in context'. "
    "You may compare multiple BASE entries if present."
    "Cite context lines like [BASE[0]]."
)

def build_prompt(question, merged):
    return f"""
{SYSTEM_PROMPT}

[CONTEXT START]
{merged['context_text']}
[CONTEXT END]

Question: {question}
Answer (one paragraph only, do not continue):
"""


# =============================
# HALLUCINATION CHECK
# =============================
def derived_from_context(num, context_numbers):
    try:
        num = float(num)
        ctx = [float(x) for x in context_numbers]
    except:
        return False

    for a in ctx:
        for b in ctx:
            
            # subtraction
            if abs(a - b) == num:
                return True

            # addition
            if a + b == num:
                return True

            # multiplication
            if a * b == num:
                return True

            # division (both directions)
            if b != 0 and a / b == num:
                return True

            if a != 0 and b / a == num:
                return True

    return False


def hallucination_check(answer, context):
    refusal_phrases = [
        "not enough data in context",
        "insufficient data",
        "cannot be determined from context"
    ]

    normalized = answer.lower().strip()
    for phrase in refusal_phrases:
        if phrase in normalized:
            return {
                "hallucinated": False,
                "missing_terms": []
            }

    # Extract numbers only (core factual risk)
    answer_numbers = set(re.findall(r"\b\d+\b", answer))
    context_numbers = set(re.findall(r"\b\d+\b", context))

    missing_numbers = set()
    
    for num in answer_numbers:
        if num not in context_numbers:
            if not derived_from_context(num, context_numbers):
                missing_numbers.add(num)

    return {
        "hallucinated": len(missing_numbers) > 0,
        "missing_terms": list(missing_numbers)
    }

# =============================
# MODEL RUNNER
# =============================
def run_model(prompt, model_key, models, context):
    start = time.time()
    text = models.generate(model_key, prompt)
    latency = time.time() - start


    for marker in ["[CONTEXT START]", "Question:"]:
        if marker in text:
            text = text.split(marker)[0].strip()
    
    halluc = hallucination_check(text, context)

    return {
        "model": model_key,
        "answer": text,
        "latency": latency,
        "hallucination": halluc
    }


def run_all_models(prompt, models, context):
    results = []
    for key in MODEL_REGISTRY:
        if models.is_ready(key):
            results.append(run_model(prompt, key, models, context))
    return results


def is_answerable(question, context_text):
    if "compare" in question.lower():
        return "BASE[0]" in context_text and "BASE[1]" in context_text
    return "Goals" in context_text


# =============================
# EVALUATION HARNESS
# =============================
def evaluate_models(test_cases, models):
    outputs = []

    for case in test_cases:
        merged = merge_context(case["baseline"], case["embeddings"])

        if not is_answerable(case["question"], merged["context_text"]):
            skipped_results = []
            for key in MODEL_REGISTRY:
                if models.is_ready(key):
                    skipped_results.append({
                        "model": key,
                        "answer": "Not enough data in context.",
                        "latency": 0,
                        "hallucination": {
                            "hallucinated": False,
                            "missing_terms": []
                        },
                        "skipped": True,
                        "reason": "Insufficient context"
                    })
            outputs.append({
                "intent": case["intent"],
                "question": case["question"],
                "results": skipped_results
            })        
            continue
        
        prompt = build_prompt(case["question"], merged)

        res = run_all_models(prompt, models, merged["context_text"])    
        outputs.append({
            "intent": case["intent"],
            "question": case["question"],
            "results": res
        })

    return outputs


models = HFModels(token=HF_TOKEN)
models.load_models(only=["llama3"])

TEST_CASES = [
    
]



results = evaluate_models(TEST_CASES, models)

for r in results:
    print("\nQUESTION:", r["question"])
    for m in r["results"]:
        print("MODEL:", m["model"])
        print("ANSWER:", m["answer"][:300])
        print("HALLUCINATION:", m["hallucination"])




models = HFModels(token=HF_TOKEN)
models.load_models()

results = evaluate_models(TEST_CASES, models)

for r in results:
    print(f"\n{'='*60}")
    print(f"QUESTION: {r['question']}")
    print(f"INTENT: {r['intent']}")
    print(f"{'='*60}")
    for m in r["results"]:
        print(f"\nMODEL: {m['model']}")
        print(f"LATENCY: {m['latency']:.2f}s")
        print(f"ANSWER: {m['answer'][:300]}")
        print(f"HALLUCINATION: {m['hallucination']}")
        if m.get('skipped'):
            print(f"⚠ SKIPPED: {m.get('reason')}")



