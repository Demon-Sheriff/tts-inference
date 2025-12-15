import modal
import os

app = modal.App("orpheus-merge-pretrain")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "huggingface_hub",
        "scipy"
    )
)

vol = modal.Volume.from_name("orpheus-cache")

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/cache": vol},
    gpu="A100", # High RAM for merging
    timeout=3600
)
def merge_and_save_pretrain():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    # Specifics for the Pretrained-FT run
    base_model_id = "canopylabs/3b-hi-pretrain-research_release"
    adapter_path = "Andy004/canopy-3b-hi-pretrain-elise-finetune" # User's adapter
    output_dir = "/cache/orpheus-merged-pretrain-vllm"
    
    print(f"Loading tokenizer from {adapter_path}...")
    # Tokenizer is saved in the adapter repo usually
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, token=os.environ["HF_TOKEN"])
    
    print(f"Loading base model {base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        token=os.environ["HF_TOKEN"],
        trust_remote_code=True
    )
    
    print(f"Resizing embeddings to {len(tokenizer)}...")
    base_model.resize_token_embeddings(len(tokenizer))
    
    print(f"Loading adapter {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path, token=os.environ["HF_TOKEN"])
    
    print("Merging weights...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    vol.commit()
    print("Merge Complete.")

@app.local_entrypoint()
def main():
    merge_and_save_pretrain.remote()
