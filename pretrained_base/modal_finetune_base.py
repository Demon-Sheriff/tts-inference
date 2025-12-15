#### **** finetuning hindi-pretrained model **** ####
import modal
import os

app = modal.App("orpheus-finetune-pretrain")

# Image with necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "bitsandbytes",
        "wandb",
        "huggingface_hub",
        "hf_transfer",
        "scipy"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

vol = modal.Volume.from_name("orpheus-cache", create_if_missing=True)

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("api-keys")
    ],
    volumes={"/cache": vol},
    gpu="A100", 
    timeout=3600 * 4 
)
def finetune_orpheus_base():
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    import re
    import wandb

    # Callback for sampling details
    class SamplingCallback(TrainerCallback):
        def __init__(self, tokenizer, model, prompt):
            self.tokenizer = tokenizer
            self.model = model
            self.prompt = prompt

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 20 == 0 and state.global_step > 0:
                print(f"Sampling at step {state.global_step}...")
                inputs = self.tokenizer(self.prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
                generated = self.tokenizer.decode(outputs[0])
                print(f"SAMPLE: {generated}")
                if wandb.run:
                    wandb.log({"sample_generation": wandb.Html(f"<p>{generated}</p>")}, step=state.global_step)

    # Configuration for Pretrained Base Model
    model_id = "canopylabs/3b-hi-pretrain-research_release" # Different base
    dataset_id = "rumik-ai/hi-elise"
    output_dir = "/cache/orpheus-finetuned-pretrain"
    hub_model_id = "canopy-3b-hi-pretrain-elise-finetune" # Different output
    
    # WandB Project specific to this comparative run
    os.environ["WANDB_PROJECT"] = "orpheus-finetune-pretrained"
    
    print(f"Loading dataset: {dataset_id}")
    ds = load_dataset(dataset_id, split="train", token=os.environ["HF_TOKEN"]).remove_columns(["audio"])
    
    print(f"Dataset Loaded. Size: {len(ds)}")
    
    # 1. Tag Extraction & Tokenizer Update
    print("Analyzing tags...")
    all_text = " ".join(ds["text"])
    tags = set(re.findall(r"<[^>]+>", all_text))
    print(f"Found {len(tags)} unique tags: {tags}")

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_TOKEN"])
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens
    special_tokens_dict = {'additional_special_tokens': list(tags)}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens to tokenizer.")

    # 2. Model Loading
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        token=os.environ["HF_TOKEN"],
        load_in_4bit=True, 
        trust_remote_code=True
    )
    
    # Resize embeddings to fit new tags
    model.resize_token_embeddings(len(tokenizer))
    
    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    
    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=2048)

    tokenized_datasets = ds.map(tokenize_function, batched=True)
    
    # 4. Training
    sample_prompt = "नमस्ते <laugh>" # Simple prompt to test tag usage
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        max_steps=100, 
        save_steps=50,
        optim="paged_adamw_8bit",
        report_to="wandb",
        push_to_hub=True,
        hub_model_id=hub_model_id,
        hub_token=os.environ["HF_TOKEN"],
        hub_private_repo=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[SamplingCallback(tokenizer, model, sample_prompt)]
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model locally...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Pushing to Hugging Face Hub...")
    try:
        trainer.push_to_hub()
        print(f"Successfully pushed to {hub_model_id}")
    except Exception as e:
        print(f"Failed to push to hub: {e}")
        
    print("Training Complete.")

@app.local_entrypoint()
def main():
    finetune_orpheus_base.remote()
