import json
from llama_cpp import Llama
from retrieve import retrieve  # Uses your retrieve.py logic

# 1. Loader
with open('./app/system_prompt.txt', 'r') as f:
    system_prompt = f.read()

# 2. Initializing...
llm = Llama(
    model_path="./base_files/beverage-safety-final-Q4_K_M.gguf",
    n_gpu_layers=-1,  # Force everything onto the M4 GPU
    n_ctx=4096,
    verbose=False
)

def get_expert_response(user_query):
    facts = retrieve(user_query, k=30)  # FIX: was 'query' (undefined), now 'user_query'

    if not facts:
        return f"I do not have verified data for '{user_query}' in my beverage database. I can only answer about specific products listed in my system."

    context_text = ""
    for res in facts:
        label = "PRODUCT DATA" if res['type'] == 'product' else "ADDITIVE SAFETY DATA"
        data = res.get('full_data', {})
        avoid_groups = data.get('avoid_if', [])

        context_text += f"\n[{label}]: {json.dumps(res.get('full_data', res))}"

        if avoid_groups:
            context_text += f"\nALERT: People who should avoid {res['name']}: {', '.join(avoid_groups)}"

    prompt = f"""[INST] <<SYS>>
{system_prompt}
<</SYS>>

CONTEXT:
{context_text}

QUESTION:
{user_query} [/INST]"""

    output = llm(prompt, max_tokens=1024, stop=["</s>"], echo=False)
    return output['choices'][0]['text']

# 3. Interactive Loop
print("\n--- Beverage Safety Expert (STRICT MODE) ---")
while True:
    query = input("\n> ")
    if query.lower() in ['exit', 'quit']:
        break
    print(get_expert_response(query))