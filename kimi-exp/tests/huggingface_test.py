import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "moonshotai/Kimi-K2-Thinking"

# 1) Load tokenizer (includes a chat template Jinja file)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)

# 2) Load model (compressed-tensors int4, custom DeepSeek-based arch)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,       # uses modeling_deepseek.py you pasted
)

model.eval()

# 3) Build a chat prompt using the repo’s chat_template.jinja
messages = [
    {
        "role": "system",
        "content": "You are Kimi, an AI assistant created by Moonshot AI."
    },
]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

# 4) Generate
with torch.no_grad():
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=256,
        do_sample=False,   # deterministic; set True + temperature for sampling
    )

# 5) Decode only the newly-generated tokens
generated = out[0, input_ids.shape[-1]:]
print(tokenizer.decode(generated, skip_special_tokens=True))
