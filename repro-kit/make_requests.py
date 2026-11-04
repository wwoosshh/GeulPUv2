"""Write repro-kit/requests.json: the models, the two settings and the 12 fixed chat requests.

Usage: python -I -X utf8 make_requests.py <out.json>
The long-context request embeds a synthetic inventory log (deterministic, no outside text).
"""
import json
import random
import sys

MODELS = [
    {
        "key": "qwen3-8b-q8_0",
        "name": "Qwen3-8B Q8_0",
        "repo": "Qwen/Qwen3-8B-GGUF",
        "file": "Qwen3-8B-Q8_0.gguf",
        "size": 8709518112,
        "sha256": "408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6",
        "license": "Apache-2.0",
        "backends": "CUDA and Vulkan",
    },
    {
        "key": "qwen3.5-9b-q4_k_m",
        "name": "Qwen3.5-9B Q4_K_M",
        "repo": "unsloth/Qwen3.5-9B-GGUF",
        "file": "Qwen3.5-9B-Q4_K_M.gguf",
        "size": 5680522464,
        "sha256": "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
        "license": "Apache-2.0",
        "backends": "CUDA only (GeulPU 0.21.0)",
    },
]

# The model of the app's own reproducibility check (Benchmark page; --builtin-check). The app finds it by size and checks the hash.
BUILTIN_CHECK = {
    "key": "deepseek-r1-distill-qwen-7b-q8_0",
    "name": "DeepSeek-R1-Distill-Qwen-7B Q8_0",
    "repo": "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
    "file": "DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf",
    "size": 8098524896,
    "sha256": "7307bfbb539fd2e8ed636315440781fed23140dc7f911f64b33418f5b978281b",
}

# App settings applied before each load (keys of /api/settings). "default" = the app's defaults; "exact" = the strict path.
CONFIGS = {
    "default": {"precision": "fast", "kv8_cache": True, "fast_attention": True, "slots": 2, "context": 8192},
    "exact": {"precision": "exact", "kv8_cache": False, "fast_attention": False, "slots": 2, "context": 8192},
}


def inventory_log():
    rng = random.Random(20261105)
    skus = [f"SKU-{n}" for n in range(1040, 1052)]
    shelves = [f"{a}{b}" for a in "ABCD" for b in range(1, 6)]
    actions = ["received", "shipped", "moved in", "counted"]
    lines = []
    day, hour = 1, 7
    for i in range(150):
        hour += rng.choice([0, 1, 1, 2])
        if hour > 18:
            day, hour = day + 1, 7
        lines.append(f"2026-10-{day:02d} {hour:02d}:{rng.choice(['00', '15', '30', '45'])} | {rng.choice(skus)} | "
                     f"{rng.choice(actions)} {rng.randint(1, 40)} units | shelf {rng.choice(shelves)} | entry {i + 1:03d}")
    return "\n".join(lines)


def user(text):
    return {"role": "user", "content": text}


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Madrid"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_files",
        "description": "Search a code repository for files whose content matches a pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression to search for"},
                "path": {"type": "string", "description": "Folder to search, relative to the repository root"},
            },
            "required": ["pattern"],
        },
    },
}
CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression and return the result.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. (3 + 4) * 12"}},
            "required": ["expression"],
        },
    },
}

GREEDY = {"temperature": 0, "seed": 7}


def req(rid, kind, messages, max_tokens=256, thinking=False, sampling=None, tools=None):
    r = {"id": rid, "kind": kind, "messages": messages, "max_tokens": max_tokens,
         "chat_template_kwargs": {"enable_thinking": thinking}}
    r.update(sampling or GREEDY)
    if tools:
        r["tools"] = tools
    return r


REQUESTS = [
    req("math-1", "math, thinking on",
        [user("A bakery sells muffins for $2.50 and scones for $3.25. On Monday it sold 48 muffins and 27 scones; on Tuesday it sold "
              "a third fewer muffins and 50% more scones than on Monday. How much more or less money did it make on Tuesday than on "
              "Monday? Show your reasoning step by step, then give the final amount.")], thinking=True),
    req("math-2", "math",
        [user("A train leaves a station at 09:40 and travels 186 km at an average speed of 72 km/h, then waits 18 minutes and travels "
              "another 95 km at 76 km/h. At what time does it arrive? Explain each step.")]),
    req("math-3", "math",
        [user("Compute 7/12 + 5/18 - 1/8 as a single fraction in lowest terms, showing the common denominator you use.")]),
    req("logic-1", "logic",
        [user("Ana, Ben and Chloe each own exactly one pet: a cat, a dog or a parrot. Ana does not own the dog. The parrot's owner "
              "is not Chloe. Ben is allergic to cats. Who owns which pet? Reason it out before answering.")]),
    req("code-1", "code",
        [user("Write a Python function `merge_intervals(intervals)` that merges overlapping closed intervals given as a list of "
              "(start, end) tuples and returns them sorted. Include a short docstring and two example calls.")]),
    req("code-2", "code",
        [user("This function should return the second largest distinct value in a list, but it fails for some inputs. Find the bug "
              "and give a fixed version.\n\n```python\ndef second_largest(xs):\n    xs.sort()\n    return xs[-2]\n```")]),
    req("tool-1", "tool call",
        [user("Is it warm enough for a picnic in Madrid this afternoon? Use the tools if needed.")], tools=[WEATHER_TOOL, CALC_TOOL]),
    req("tool-2", "tool call",
        [{"role": "system", "content": "You are a coding assistant working in the repository /work/app. Use the tools to look things up."},
         user("Where is the function that parses the configuration file defined? I think it is called load_config.")],
        tools=[SEARCH_TOOL, CALC_TOOL]),
    req("korean-1", "Korean",
        [user("하늘이 파란 까닭을 초등학생에게 설명하듯 세 문장으로 써 주세요.")]),
    req("spanish-1", "Spanish, sampled",
        [user("Escribe un poema breve de cuatro versos sobre el mar en invierno.")],
        sampling={"temperature": 0.8, "top_p": 0.95, "seed": 1234}),
    req("long-1", "long context (~4k tokens)",
        [user("Here is a warehouse log. Each line has a date, a product code, an action with a number of units, a shelf and an "
              "entry number.\n\n" + inventory_log() + "\n\nFrom this log only: on which shelf was SKU-1047 last received, and how "
              "many units of SKU-1047 were shipped in total? Answer briefly.")], max_tokens=128),
    req("chat-1", "multi-turn, sampled",
        [user("I want to learn to cook. What are three simple dishes to start with?"),
         {"role": "assistant", "content": "Good choices are scrambled eggs, a tomato pasta sauce and a vegetable stir-fry. Each "
                                          "teaches a basic technique: gentle heat, simmering and quick high-heat cooking."},
         user("Let's start with the pasta sauce. Give me a short ingredient list and the steps.")],
        sampling={"temperature": 0.7, "top_p": 0.9, "seed": 42}),
]


def main():
    out = {
        "kit": "1",
        "geulpu": "0.21.0",
        "about": "Fixed requests of the GeulPU reproducibility kit. Each request is sent to /v1/chat/completions with the "
                 "\"geulpu\" extension (logit_hash, prompt_tokens); the kit compares the prompt tokens, the generated tokens and "
                 "every step's logit hash with expected.json.",
        "models": MODELS,
        "builtin_check": BUILTIN_CHECK,
        "configs": CONFIGS,
        "requests": REQUESTS,
    }
    with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{len(REQUESTS)} requests, {len(MODELS)} models, {len(CONFIGS)} configs -> {sys.argv[1]}")


if __name__ == "__main__":
    main()
