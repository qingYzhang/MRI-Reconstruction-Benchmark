import math

import torch
import torch.nn as nn


def make_coordinate_grid(
    height,
    width,
    device=None,
    dtype=torch.float32,
):
    """
    Source-compatible NeRP 2-D coordinate grid.

    Original code:
        grid_y, grid_x = meshgrid(
            linspace(0,1,h),
            linspace(0,1,w)
        )
        grid = stack([grid_y, grid_x], -1)

    Returns:
        [H, W, 2]
    """

    y = torch.linspace(
        0.0,
        1.0,
        steps=height,
        device=device,
        dtype=dtype,
    )

    x = torch.linspace(
        0.0,
        1.0,
        steps=width,
        device=device,
        dtype=dtype,
    )

    grid_y, grid_x = torch.meshgrid(
        y,
        x,
        indexing="ij",
    )

    return torch.stack(
        [grid_y, grid_x],
        dim=-1,
    )


class GaussianFourierEncoder(nn.Module):
    """
    Source-faithful NeRP Gaussian Fourier
    positional encoding.

        B ~ N(0, sigma^2)

        embedding(x) =
            [
                sin(2*pi*x*B^T),
                cos(2*pi*x*B^T)
            ]

    MRI settings from paper:
        coordinates_size = 2
        embedding_size   = 256
        sigma            = 3
    """

    def __init__(
        self,
        coordinates_size=2,
        embedding_size=256,
        scale=3.0,
        seed=None,
    ):
        super().__init__()

        if seed is None:
            B = torch.randn(
                embedding_size,
                coordinates_size,
            )
        else:
            generator = torch.Generator(
                device="cpu"
            )
            generator.manual_seed(seed)

            B = torch.randn(
                embedding_size,
                coordinates_size,
                generator=generator,
            )

        B = B * float(scale)

        #
        # Original repo saves encoder.B with
        # the model. Using a buffer gives us
        # the same behavior while letting
        # .to(device) work normally.
        #
        self.register_buffer(
            "B",
            B,
        )

        self.coordinates_size = int(
            coordinates_size
        )

        self.embedding_size = int(
            embedding_size
        )

        self.scale = float(
            scale
        )

    @property
    def output_size(self):
        return (
            2
            * self.embedding_size
        )

    def forward(self, coordinates):

        if coordinates.shape[-1] != (
            self.coordinates_size
        ):
            raise ValueError(
                f"Expected coordinate dimension "
                f"{self.coordinates_size}, "
                f"got {coordinates.shape}"
            )

        projected = (
            2.0
            * math.pi
            * coordinates
        ) @ self.B.t()

        return torch.cat(
            [
                torch.sin(projected),
                torch.cos(projected),
            ],
            dim=-1,
        )
