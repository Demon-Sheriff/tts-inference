import modal
import os

app = modal.App("dataset-inspector")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("datasets", "huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def inspect_dataset():
    from datasets import load_dataset
    
    dataset_id = "rumik-ai/hi-elise"
    
    print(f"Loading dataset: {dataset_id} (TEXT ONLY)")
    # Load WITHOUT audio to avoid decoding issues
    ds = load_dataset(dataset_id, split="train", token=os.environ["HF_TOKEN"])
    
    # Remove audio column immediately
    if 'audio' in ds.column_names:
        print("Removing audio column to avoid decoding errors...")
        ds = ds.remove_columns(['audio'])
    
    print(f"\n{'='*60}")
    print(f"DATASET STRUCTURE")
    print(f"{'='*60}")
    print(f"Total samples: {len(ds)}")
    print(f"Columns: {ds.column_names}")
    print(f"Features: {ds.features}")
    
    print(f"\n{'='*60}")
    print(f"SAMPLE EXAMPLES (First 5)")
    print(f"{'='*60}")
    
    for i in range(min(5, len(ds))):
        print(f"\n--- SAMPLE {i+1} ---")
        sample = ds[i]
        
        # Show text
        if 'text' in sample:
            text = sample['text']
            print(f"TEXT: {text[:200]}{'...' if len(text) > 200 else ''}")
            
            # Count tags
            import re
            tags = re.findall(r'<[^>]+>', text)
            if tags:
                print(f"TAGS FOUND: {set(tags)}")
            else:
                print(f"TAGS FOUND: None")
        
        # Show audio info
        if 'audio' in sample:
            audio_info = sample['audio']
            if isinstance(audio_info, dict):
                print(f"AUDIO: {audio_info.get('path', 'N/A')}")
                if 'sampling_rate' in audio_info:
                    print(f"  - Sampling rate: {audio_info['sampling_rate']} Hz")
                if 'array' in audio_info:
                    duration = len(audio_info['array']) / audio_info.get('sampling_rate', 16000)
                    print(f"  - Duration: ~{duration:.2f}s")
    
    print(f"\n{'='*60}")
    print(f"TAG ANALYSIS")
    print(f"{'='*60}")
    
    all_text = " ".join(ds["text"])
    all_tags = re.findall(r'<[^>]+>', all_text)
    unique_tags = set(all_tags)
    
    print(f"Unique tags: {len(unique_tags)}")
    print(f"Tags: {sorted(unique_tags)}")
    
    # Tag distribution
    from collections import Counter
    tag_counts = Counter(all_tags)
    print(f"\nMost common tags:")
    for tag, count in tag_counts.most_common(10):
        print(f"  {tag}: {count}")

@app.local_entrypoint()
def main():
    inspect_dataset.remote()
