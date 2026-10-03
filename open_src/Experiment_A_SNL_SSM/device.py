import torch
import gc
import subprocess

def get_free_gpu():
    try:
        # 查询显存占用
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE, text=True
        )
        memory_used = [int(x) for x in result.stdout.strip().split('\n')]
        free_gpu = min(range(len(memory_used)), key=lambda i: memory_used[i])
        return free_gpu
    except Exception:
        return 0  # 出错默认 0 号
    
def get_device():
    if torch.cuda.is_available():
        gpu_id = get_free_gpu()
        device = torch.device(f"cuda:{gpu_id}")
        use_gpu = True
        #print(f"GPU: cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
        use_gpu = False
        print("use cpu")
    return device, use_gpu


def delete_tensor(*tensors, use_gpu):
    """
    删除任意数量的 tensor，并根据需要清理 GPU 缓存。

    参数：
        *tensors : 可变数量的 torch.Tensor，例如:
                   delete_tensor(t1, t2, t3)
        use_gpu : bool，是否正在使用 GPU（是否清空缓存）
    """

    for t in tensors:
        if isinstance(t, torch.Tensor):
            print(f"Deleting tensor on device: {t.device} | shape: {tuple(t.shape)}")
        else:
            print("Warning: One object is not a torch.Tensor")

        del t

    gc.collect()

    if use_gpu:
        torch.cuda.empty_cache()
        print("GPU cache cleaned.")

    print("All specified tensors deleted.\n")
