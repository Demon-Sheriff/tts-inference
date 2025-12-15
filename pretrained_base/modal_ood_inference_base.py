import modal
import os

app = modal.App("orpheus-ood-pretrain")

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
        "hf_transfer"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

vol = modal.Volume.from_name("orpheus-cache")

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/cache": vol},
    gpu="T4"
)
def run_pretrain_inference(prompts):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    # Pretrained Base + New Adapter
    base_model_id = "canopylabs/3b-hi-pretrain-research_release"
    adapter_id = "Andy004/canopy-3b-hi-pretrain-elise-finetune"
    
    print(f"Loading tokenizer {base_model_id}...")
    # Load from adapter to get new tokens
    tokenizer = AutoTokenizer.from_pretrained(adapter_id, token=os.environ["HF_TOKEN"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading base model {base_model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        token=os.environ["HF_TOKEN"],
        load_in_4bit=True,
        trust_remote_code=True
    )
    
    # Resize embeddings - critical!
    print(f"Resizing embeddings to {len(tokenizer)}...")
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"Loading adapter {adapter_id}...")
    model = PeftModel.from_pretrained(model, adapter_id, token=os.environ["HF_TOKEN"])
    
    print("Generating OOD samples...")
    results = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=60, 
                do_sample=True, 
                temperature=0.7,
                repetition_penalty=1.2 # Help with loops
            )
            
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        results.append(generated_text)
        
    return results

@app.local_entrypoint()
def main():
    # Robust OOD prompts (same as benchmark)
    test_prompts = [
        "नमस्ते, आज का दिन कैसा है?",
        "मुझे विश्वास नहीं हो रहा कि तुमने ऐसा किया। <sigh>",
        "<romantic music playing> वह धीरे से उसके पास आया और बोला,",
        "रुको! वहां कौन है? <gasps>",
        "खाना बहुत स्वादिष्ट था। <burps> अरे माफ़ करना!",
        "This is really funny. <laughs>" # Code mixing
    ]
    
    print("Running Inference on Pretrained-FT Model...")
    results = run_pretrain_inference.remote(test_prompts)
    
    print("\n--- RESULTS ---")
    for res in results:
        print(f"{res}\n{'-'*20}")
