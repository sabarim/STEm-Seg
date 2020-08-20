import torch


def get_nb_embedding_dims(mode):
    if mode in ("xy", "ff"):
        return 2
    elif mode in ("xyt", "xyf"):
        return 3
    elif mode in ("xytf", "xyff"):
        return 4
    elif mode in ("xytff", "xyfff"):
        return 5
    else:
        raise ValueError("Invalid experimental embedding mode: {}".format(mode))


def get_nb_free_dims(mode):
    if mode in ("xyf", "xytf"):
        return 1
    elif mode in ("xyff", "xytff"):
        return 2
    elif mode == "xyfff":
        return 3
    else:
        return 0


@torch.no_grad()
def creat_spatiotemporal_grid(height, width, time, t_scale, dtype=torch.float32, device="cpu"):
    # returns [tx, ty, txy, y, x]
    x_abs = max(1., width / float(height))
    y_abs = max(1., height / float(width))

    # torch.linspace does not work with float16, so create the tensors using float32 and then cast to appropriate dtype
    x = torch.linspace(-x_abs, x_abs, width, dtype=torch.float32, device=device).to(dtype=dtype)
    y = torch.linspace(-y_abs, y_abs, height, dtype=torch.float32, device=device).to(dtype=dtype)
    t = torch.linspace(-t_scale, t_scale, time, dtype=torch.float32, device=device).to(dtype=dtype)

    t, y, x = torch.meshgrid(t, y, x)

    return t, y, x


def add_spatiotemporal_offset(embeddings, time_scale, mode):
    N, C, T, H, W = embeddings.shape
    t, y, x = creat_spatiotemporal_grid(H, W, T, time_scale, embeddings.dtype, embeddings.device)

    if mode == "x":
        with torch.no_grad():
            grid = x.unsqueeze(0)

        return embeddings + grid.detach()

    elif mode == "xyf":
        with torch.no_grad():
            zeros = torch.zeros_like(x)
            grid = torch.stack((y, x, zeros), dim=0)
            grid = grid.unsqueeze(0).expand(N, -1, -1, -1, -1)  # [N, 3, T, H, W]

        return embeddings + grid.detach()

    elif mode == "ff":
        return embeddings

    elif mode == "xytf":
        with torch.no_grad():
            zeros = torch.zeros_like(x)
            grid = torch.stack((t, y, x, zeros), dim=0)
            grid = grid.unsqueeze(0).expand(N, -1, -1, -1, -1)  # [N, 4, T, H, W]

        return embeddings + grid.detach()

    elif mode == "xytff":
        with torch.no_grad():
            zeros = torch.zeros_like(x)
            grid = torch.stack((t, y, x, zeros, zeros), dim=0)
            grid = grid.unsqueeze(0).expand(N, -1, -1, -1, -1)  # [N, 5, T, H, W]

        return embeddings + grid.detach()

    elif mode == "xyff":
        with torch.no_grad():
            zeros = torch.zeros_like(x)
            grid = torch.stack((y, x, zeros, zeros), dim=0)
            grid = grid.unsqueeze(0).expand(N, -1, -1, -1, -1)  # [N, 4, T, H, W]

        return embeddings + grid.detach()

    elif mode == "xyfff":
        with torch.no_grad():
            zeros = torch.zeros_like(x)
            grid = torch.stack((y, x, zeros, zeros, zeros), dim=0)
            grid = grid.unsqueeze(0).expand(N, -1, -1, -1, -1)  # [N, 5, T, H, W]

        return embeddings + grid.detach()

    elif mode == "xyffff":
        with torch.no_grad():
            zeros = torch.zeros_like(x)
            grid = torch.stack((y, x, zeros, zeros, zeros, zeros), dim=0)
            grid = grid.unsqueeze(0).expand(N, -1, -1, -1, -1)  # [N, 6, T, H, W]

        return embeddings + grid.detach()

    elif mode == "xy":
        with torch.no_grad():
            grid = torch.stack((y, x), dim=0)
            grid = grid.unsqueeze(0).expand(N, -1, -1, -1, -1)  # [N, 2, T, H, W]

        return embeddings + grid.detach()

    elif mode == "xyt":
        with torch.no_grad():
            grid = torch.stack((t, y, x), dim=0)
            grid = grid.unsqueeze(0).expand(N, -1, -1, -1, -1)  # [N, 3, T, H, W]

        return embeddings + grid.detach()

    else:
        raise ValueError("Invalid experimental embedding mode: {}".format(mode))
