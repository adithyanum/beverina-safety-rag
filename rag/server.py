import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
from retrieve import retrieve 

app = FastAPI()

# Enable CORS so the frontend can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- YOUR EXISTING LOGIC ---
print("--- Loading Model... ---")
# 1. Loader
with open('./app/system_prompt.txt', 'r') as f:
    system_prompt = f.read()

# 2. Initializing
llm = Llama(
    model_path="./base_files/beverage-safety-final-Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False
)

class UserQuery(BaseModel):
    query: str

@app.post("/api/chat")
async def chat_endpoint(payload: UserQuery):
    user_query = payload.query
    
    # Retrieval
    facts = retrieve(user_query, k=30)

    if not facts:
        return {"response": "I don't have verified data for this in my beverage database. Please refer to the official product website or FSSAI sources for accurate information."}
    
    context_text = ""
    for res in facts:
        label = "PRODUCT DATA" if res['type'] == 'product' else "ADDITIVE SAFETY DATA"
        # Handle cases where full_data might be missing or complex
        data_content = res.get('full_data', res)
        avoid_groups = data_content.get('avoid_if', []) if isinstance(data_content, dict) else []

        context_text += f"\n[{label}]: {json.dumps(data_content)}"
        if avoid_groups:
            context_text += f"\nALERT: People who should avoid {res.get('name', 'Unknown')}: {', '.join(avoid_groups)}"

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_prompt}
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    CONTEXT:
    {context_text}

    QUESTION:
    {user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    output = llm(prompt, max_tokens=1024, stop=["<|eot_id|>"], echo=False)
    
    return {"response": output['choices'][0]['text']}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)