import torch
# from transformers import BitsAndBytesConfig
# import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(0)
text = "Hello my name is"
inputs = tokenizer(text, return_tensors="pt").to(0)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# bnb_config = BitsAndBytesConfig(
#   load_in_4bit=True,
#   bnb_4bit_compute_dtype=torch.float16
# )
   
# pipeline = transformers.pipeline(
#   task="text-generation",
#   model=model_id,
#   model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True,  "quantization_config": bnb_config},
# )


# messages = [{"role": "user", "content": "Explain what is electronic money like I'm a 7 years old kid."}]
# prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
# print(outputs[0]["generated_text"])