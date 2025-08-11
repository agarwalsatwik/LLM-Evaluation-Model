import os
from huggingface_hub import InferenceClient
import model

client = InferenceClient(
    provider="together",
    api_key=os.environ["HF_TOKEN"],
)

# ===== models ===== 
llama_70B = "meta-llama/Llama-3.3-70B-Instruct"
mistral = "mistralai/Mistral-7B-Instruct-v0.3"
llama_3B = "meta-llama/Llama-3.2-3B-Instruct"

models = [llama_70B, mistral, llama_3B]

# ===== prompt =====
prompt = "Compare reinforcement learning and deep learning in 1000 words."

map = {}
responses = []

for mdl in models:
    response = client.chat.completions.create(model=mdl, messages=[{"role": "user", "content": prompt}]).choices[0].message.content
    responses.append(response)
    score = model.scoreModel(response, 1000)
    print(mdl,": ",score,"\n")
    map[mdl] = score

top_model = models[0]
top_score = map[top_model]

for key in map:
    if score < map[key]:
        top_model = key
        top_score  = map[top_model]

print(top_model,"is the best model, with score:",top_score)

for i in range(5):
    print("\n")

for res in responses:
    print("="*80,"LLM RESPONSE","="*80,"\n")
    print(res)
