import torch
import triton
import triton.language as tl
from triton.testing import do_bench

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def get_cuda_autotune_config():
    # (BM, BN, num_warps, num_stages, swizzle_m)
    specs = [
        (128, 128, 4, 4),
        (128, 64, 4, 4),
        (64, 128, 4, 4),
        (64, 64, 2, 5),
        (128, 256, 8, 3),
        (256, 128, 8, 3),
        (128, 32, 4, 4),
        (32, 128, 4, 4),
        (128, 128, 4, 4),
        (128, 64, 4, 4),
        (64, 128, 4, 4),
        (64, 64, 2, 5),
        (128, 256, 8, 3),
        (256, 128, 8, 3),
        (128, 32, 4, 4),
        (32, 128, 4, 4),
    ]
    return [
        triton.Config({
            'BLOCK_SIZE_M': bm,
            'BLOCK_SIZE_N': bn
        },
                      num_warps=warps,
                      num_stages=stages) for (bm, bn, warps, stages) in specs
    ]

def validate_results(*args, **kwargs):
    a = args[0]['a_ptr']
    b = args[0]['b_ptr']
    c = args[0]['c_ptr']
    torch.cuda.synchronize(c.device)
    triton_output = c
    torch_output = torch.matmul(a, b)
    if not torch.allclose(triton_output, torch_output, atol=1e-4, rtol=1e-5):
        print("Post hook: Test failed! ❌")
    else:
        print("Post hook: Test passed! ✅")

# Filter configurations depending on the matrix sizes
def _prune_configs(configs, named_args, **kwargs):
    M, N = named_args['M'], named_args['N']
    pruned_configs = [
        c for c in configs if \
            (c.kwargs['BLOCK_SIZE_M']<=M) and \
            (c.kwargs['BLOCK_SIZE_N']<=N)
    ]
    if len(pruned_configs) == 0:
        raise Exception("No valid configurations found for the given matrix size.")
    return pruned_configs

@triton.autotune(configs=get_cuda_autotune_config(),
                 key=['M', 'N', 'K'],
                 prune_configs_by={"early_config_prune": _prune_configs},
                 post_hook=validate_results,
                 cache_results=True)

@triton.jit
def gemv_kernel(a_ptr, b_ptr, c_ptr, M, N, BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr):
    """
    C = A x B
    A: (M, N) float32
    B: (N) float32
    C: (M) float32
    """

    pid_m = tl.program_id(axis=0)

    # determine the m-tile
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M # modulo prevents overflow
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + offs_am[:, None] * N + offs_n[None, :]    # 1 column with ptrs to A rows
    b_ptrs = b_ptr + offs_n

    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    # split summation per-element along N
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        a = tl.load(a_ptrs, mask=offs_n[None, :] < N - n * BLOCK_SIZE_N, other=0.0)    # [None, :] broadcasts 1d array to 2d
        b = tl.load(b_ptrs, mask=offs_n < N - n * BLOCK_SIZE_N, other=0.0)
        accumulator += tl.sum(a * b[None, :], axis=1)
        a_ptrs += BLOCK_SIZE_N
        b_ptrs += BLOCK_SIZE_N

    c = accumulator

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = c_ptr + (offs_cm)
    c_mask = (offs_cm < M)
    tl.store(c_ptrs, c, mask=c_mask)

def check_and_launch_gemv(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, N = a.shape

    c = torch.empty((N), device=a.device, dtype=torch.float32)

    # Number of blocks = ceil(M / BLOCK_SIZE_M) * ceil(N / BLOCK_SIZE_N)
    # Number of threads per block: num_warps * 32
    # Each thread computes ≈ (BLOCK_SIZE_M * BLOCK_SIZE_N) / (num_warps * 32) elements of C
    # Define a 1-D grid of blocks
    grid = lambda META: ( \
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']))
    gemv_kernel[grid](a, b, c, M, N)
    return c
