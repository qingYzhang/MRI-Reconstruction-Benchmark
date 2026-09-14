import sys
from pathlib import Path

import torch


CODE_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(CODE_ROOT)
)


from methods.nerp.encoding import (
    GaussianFourierEncoder,
    make_coordinate_grid,
)

from methods.nerp.network import (
    NeRP2D,
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def count_trainable_parameters(
    model,
):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    torch.manual_seed(0)

    print("=" * 80)
    print("NERP NETWORK UNIT TEST")
    print("=" * 80)

    print(
        "device:",
        DEVICE
    )

    encoder = (
        GaussianFourierEncoder(
            coordinates_size=2,
            embedding_size=256,
            scale=3.0,
            seed=0,
        )
    )

    model = NeRP2D(
        encoder=encoder,
        depth=8,
        width=512,
        output_size=1,
        w0=30.0,
    ).to(DEVICE)

    grid = make_coordinate_grid(
        height=32,
        width=40,
        device=DEVICE,
    )

    embedding = (
        model.encoder(
            grid
        )
    )

    output = model(
        grid
    )

    print(
        "grid shape:",
        tuple(
            grid.shape
        )
    )

    print(
        "grid min:",
        grid.min().item()
    )

    print(
        "grid max:",
        grid.max().item()
    )

    print(
        "B shape:",
        tuple(
            model.encoder.B.shape
        )
    )

    print(
        "B std:",
        model.encoder.B.std().item()
    )

    print(
        "embedding shape:",
        tuple(
            embedding.shape
        )
    )

    print(
        "output shape:",
        tuple(
            output.shape
        )
    )

    print(
        "trainable parameters:",
        count_trainable_parameters(
            model
        )
    )

    print(
        "finite embedding:",
        torch.isfinite(
            embedding
        ).all().item()
    )

    print(
        "finite output:",
        torch.isfinite(
            output
        ).all().item()
    )

    #
    # Gradient test.
    #
    target = torch.zeros_like(
        output
    )

    loss = torch.mean(
        (
            output
            - target
        ) ** 2
    )

    loss.backward()

    finite_gradients = all(
        p.grad is None
        or torch.isfinite(
            p.grad
        ).all().item()
        for p in model.parameters()
    )

    print(
        "loss:",
        loss.item()
    )

    print(
        "finite gradients:",
        finite_gradients
    )

    #
    # Encoder reproducibility.
    #
    encoder2 = (
        GaussianFourierEncoder(
            coordinates_size=2,
            embedding_size=256,
            scale=3.0,
            seed=0,
        )
    )

    same_B = torch.equal(
        encoder2.B.cpu(),
        model.encoder.B.cpu(),
    )

    print(
        "same seed -> same B:",
        same_B
    )


if __name__ == "__main__":
    main()
