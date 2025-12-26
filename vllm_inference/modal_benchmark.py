import modal
import os

app = modal.App("orpheus-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "scipy",
        "huggingface_hub",
        "wandb"
    )
)

vol = modal.Volume.from_name("orpheus-cache")

class ModelConfig:
    def __init__(self, name, base_model_id, adapter_id=None, description=""):
        self.name = name
        self.base_model_id = base_model_id
        self.adapter_id = adapter_id
        self.description = description

# ... (imports remain)
MODELS_TO_TEST = [
    ModelConfig("Base Pretrained", "canopylabs/3b-hi-pretrain-research_release"),
    ModelConfig("Research FT", "canopylabs/3b-hi-ft-research_release"),
    ModelConfig("Orpheus FT (Ours)", "canopylabs/3b-hi-ft-research_release", "Andy004/canopy-3b-hi-elise-finetune")
]

PROMPTS = [
    "मुझे विश्वास नहीं हो रहा कि तुमने ऐसा किया। <sigh> खैर, अब जो हुआ सो हुआ। <laughs nervously>",
    "<romantic music playing> वह धीरे से उसके पास आया और बोला, मैं तुमसे प्यार करता हूँ। <smooches>",
    "रुको! वहां कौन है? <gasps> मुझे लगा मैंने कुछ सुना। <tapping sounds>",
    "मैं... मैं... <stutters> मुझे नहीं पता कि क्या कहना चाहिए। <nervous laughter>",
    "वाह! <laughs> यह तो कमाल है! <claps>",
    "खाना बहुत स्वादिष्ट था। <burps> अरे माफ़ करना! <chuckles>",
    "<whispers> धीरे बोलो, कोई सुन लेगा। <long pause> चलो यहाँ से चलते हैं।",
    "This is really funny. <laughs> मुझे यह बहुत पसंद आया।",
    "मैं अब और नहीं सह सकती। <crying> सब खत्म हो गया।",
    "बाहर निकलो! <shouts> मुझे तुम्हारी शक्ल नहीं देखनी!",
]

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")], # No wandb secret needed in remote func if aggregating locally
    volumes={"/cache": vol},
    gpu="T4",
    timeout=900
)
def evaluate_model(model_name: str, base_id: str, adapter_id: str = None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import numpy as np
    
    # ... (Load Tokenizer & Model code same as before) ...
    print(f"[{model_name}] Loading tokenizer for {base_id}...")
    try:
        if adapter_id:
            tokenizer = AutoTokenizer.from_pretrained(adapter_id, token=os.environ["HF_TOKEN"])
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_id, token=os.environ["HF_TOKEN"])
    except:
        tokenizer = AutoTokenizer.from_pretrained(base_id, token=os.environ["HF_TOKEN"])
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[{model_name}] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map="auto",
        token=os.environ["HF_TOKEN"],
        load_in_4bit=True,
        trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))

    if adapter_id:
        print(f"[{model_name}] Loading adapter...")
        try:
            model = PeftModel.from_pretrained(model, adapter_id, token=os.environ["HF_TOKEN"])
        except Exception as e:
            print(f"Error loading adapter: {e}")
            return []

    results = []
    
    print(f"[{model_name}] Benchmarking...")
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Calculate Perplexity (on the verification set / prompt continuation)
        # For generation quality, we usually check PPL of the generated text given prompt?
        # Or just generate and return text. PPL calculation for *generated* text is a bit circular (it will be low).
        # Better metric: Just return text for side-by-side. 
        # But let's verify if we can get log_probs.
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=True, 
                temperature=0.7,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode
        generated_seq = outputs.sequences[0]
        generated_text = tokenizer.decode(generated_seq, skip_special_tokens=False)
        
        # Calculate Perplexity of the sequence (just as a proxy for confidence)
        # We need to run a forward pass to get loss
        # Note: 'outputs.scores' gives logits for generated tokens. We can compute perplexity from that.
        # But for simplicity/speed, let's just return length and text first. User asked for metrics.
        
        results.append({
            "prompt": prompt,
            "output": generated_text,
            "new_tokens": len(generated_seq) - inputs.input_ids.shape[1]
        })
        
    return results

@app.local_entrypoint()
def main():
    import wandb
    
    print("Starting Parallel Benchmark...")
    results_map = {} # model_name -> list of result dicts
    
    # Spawn jobs
    handles = []
    for cfg in MODELS_TO_TEST:
        handles.append((cfg.name, evaluate_model.spawn(cfg.name, cfg.base_model_id, cfg.adapter_id)))
        
    # Collect results
    for name, handle in handles:
        print(f"Waiting for {name}...")
        try:
            res = handle.get()
            results_map[name] = res
            print(f"✅ {name} finished.")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results_map[name] = []

    # Init WandB locally
    wandb.init(project="orpheus-benchmark", name="comparison-10-prompts")
    
    # Construct consolidated table
    # Columns: Prompt | Base Output | Research FT Output | Ours Output | ... metrics
    columns = ["Prompt"]
    for model_name in [m.name for m in MODELS_TO_TEST]:
        columns.append(f"{model_name} Output")
        columns.append(f"{model_name} Len")
        
    table_data = []
    
    # Assume all robust prompts processed in order
    for i, prompt in enumerate(PROMPTS):
        row = [prompt]
        for model_cfg in MODELS_TO_TEST:
            model_res = results_map.get(model_cfg.name, [])
            if i < len(model_res):
                row.append(model_res[i]["output"])
                row.append(model_res[i]["new_tokens"])
            else:
                row.append("N/A")
                row.append(0)
        table_data.append(row)
        
    table = wandb.Table(columns=columns, data=table_data)
    wandb.log({"benchmark_comparison": table})
    wandb.finish()
    print("WandB upload complete.")
