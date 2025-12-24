import modal
app = modal.App("test_gpu")

@app.function(gpu=["H100"])
def test_gpus():
    import subprocess
    try:
        subprocess.run(["nvidia-smi", "--list-gpus"])
    except Exception as e:
        print("NVIDIA GPU not found")

