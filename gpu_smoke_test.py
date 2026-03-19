import time
import torch

def main():
    print("torch.__version__ =", torch.__version__)
    print("cuda available     =", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit(2)

    print("device count       =", torch.cuda.device_count())
    print("device name        =", torch.cuda.get_device_name(0))

    # warmup
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    b = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()

    t0 = time.time()
    c = a @ b
    torch.cuda.synchronize()
    t1 = time.time()

    print("matmul done. shape =", tuple(c.shape))
    print("elapsed (s)        =", t1 - t0)
    print("c mean            =", float(c.mean().item()))

if __name__ == "__main__":
    main()
