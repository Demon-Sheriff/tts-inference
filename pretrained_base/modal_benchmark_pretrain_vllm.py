import modal
import time
import time
from pretrained_base.modal_inference_vllm_pretrain import app, ModelInferencePretrain

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

@app.local_entrypoint()
def benchmark():
    import wandb
    
    print("Starting Benchmark: Pretrained-Base-FT on vLLM (A100)...")
    
    # Warmup
    ModelInferencePretrain().generate.remote(prompts=PROMPTS[:1])
    
    # Run
    start_time = time.time()
    results = ModelInferencePretrain().generate.remote(prompts=PROMPTS)
    end_time = time.time()
    
    total_chars = sum(len(r) for r in results)
    duration = end_time - start_time
    
    print(f"Time: {duration:.2f}s")
    print(f"Throughput: {total_chars/duration:.2f} chars/s")
    
    run = wandb.init(project="orpheus-benchmark", name="vllm-pretrain-ft-a100")
    
    table_data = []
    for p, r in zip(PROMPTS, results):
        table_data.append([p, r, len(r)])
        
    table = wandb.Table(columns=["Prompt", "Pretrained-FT Output", "Length"], data=table_data)
    run.log({"vllm_pretrain_results": table})
    run.finish()
    print("Uploaded to WandB.")
