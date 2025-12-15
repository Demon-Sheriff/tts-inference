import modal

app = modal.App("orpheus-vllm-pretrain")

# Same image setup as the working A100 vLLM script
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm", 
        "huggingface_hub", 
        "hf_transfer"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

vol = modal.Volume.from_name("orpheus-cache")

@app.cls(
    image=image,
    gpu="A100", # Using A100 for stability
    volumes={"/cache": vol},
    scaledown_window=300
)
class ModelInferencePretrain:
    @modal.enter()
    def load_model(self):
        from vllm import LLM
        
        # Path to the NEW merged model (Pretrained Base + FT Adapter)
        model_path = "/cache/orpheus-merged-pretrain-vllm"
        
        print(f"Loading vLLM engine from {model_path}...")
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            dtype="float16"
        )
        print("vLLM engine loaded.")

    @modal.method()
    def generate(self, prompts: list[str]):
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=100,
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
            
        return results
