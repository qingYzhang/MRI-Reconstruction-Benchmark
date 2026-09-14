import math

import torch
import torch.nn as nn


class SirenLayer(nn.Module):
    """
    Direct port of NeRP SirenLayer.

    Source behavior:
        first layer:
            weight ~ U(-1/in_f, 1/in_f)

        other layers:
            weight ~ U(
                -sqrt(6/in_f)/w0,
                 sqrt(6/in_f)/w0
            )

        activation:
            sin(w0 * linear(x))

        final layer:
            linear only
    """

    def __init__(
        self,
        in_features,
        out_features,
        w0=30.0,
        is_first=False,
        is_last=False,
    ):
        super().__init__()

        self.in_features = int(
            in_features
        )

        self.w0 = float(
            w0
        )

        self.is_first = bool(
            is_first
        )

        self.is_last = bool(
            is_last
        )

        self.linear = nn.Linear(
            in_features,
            out_features,
        )

        self._initialize_weights()

    def _initialize_weights(self):

        if self.is_first:

            bound = (
                1.0
                / self.in_features
            )

        else:

            bound = (
                math.sqrt(
                    6.0
                    / self.in_features
                )
                / self.w0
            )

        #
        # Match original NeRP:
        # only overwrite weights.
        # PyTorch default bias initialization
        # is left unchanged.
        #
        with torch.no_grad():

            self.linear.weight.uniform_(
                -bound,
                bound,
            )

    def forward(self, x):

        x = self.linear(x)

        if self.is_last:
            return x

        return torch.sin(
            self.w0 * x
        )


class NeRPSiren(nn.Module):
    """
    NeRP MRI implicit image representation.

    Paper MRI configuration:
        depth  = 8
        width  = 512
        input  = 512 Fourier features
        output = 1 intensity
    """

    def __init__(
        self,
        input_size=512,
        output_size=1,
        depth=8,
        width=512,
        w0=30.0,
    ):
        super().__init__()

        if depth < 2:
            raise ValueError(
                "depth must be >= 2"
            )

        layers = [
            SirenLayer(
                input_size,
                width,
                w0=w0,
                is_first=True,
            )
        ]

        for _ in range(
            1,
            depth - 1,
        ):

            layers.append(
                SirenLayer(
                    width,
                    width,
                    w0=w0,
                )
            )

        layers.append(
            SirenLayer(
                width,
                output_size,
                w0=w0,
                is_last=True,
            )
        )

        self.model = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.model(x)


class NeRP2D(nn.Module):
    """
    Complete coordinate -> image NeRP model.
    """

    def __init__(
        self,
        encoder,
        depth=8,
        width=512,
        output_size=1,
        w0=30.0,
    ):
        super().__init__()

        self.encoder = encoder

        self.network = NeRPSiren(
            input_size=
                encoder.output_size,
            output_size=
                output_size,
            depth=
                depth,
            width=
                width,
            w0=
                w0,
        )

    def forward(self, coordinates):

        embedding = self.encoder(
            coordinates
        )

        return self.network(
            embedding
        )
