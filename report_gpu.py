import torch
import time
import fire


def gpu_info():
    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i) // (1024 ** 2)} MB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(i) // (1024 ** 2)} MB")
    else:
        print("CUDA is not available.")


def run_linear(N=10000, size=(128, 128)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    linear = torch.nn.Linear(size[1], size[0]).to(device)
    for _ in range(N):
        x = torch.randn(size, device=device)
        y = linear(x)

    del x, y, linear
    torch.cuda.empty_cache()


def main(interval=3600, count=10000):
    while True:
        print(f"\n--- GPU Info Report at {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
        gpu_info()
        print(f"Running {count} linear ops...")
        run_linear(count)
        print(f"Done. Sleeping for {interval} seconds.\n")
        time.sleep(interval)


if __name__ == "__main__":
    fire.Fire(main)
