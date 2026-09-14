import time

import torch
import torch.nn.functional as F


from methods.nerp.encoding import (
    GaussianFourierEncoder,
    make_coordinate_grid,
)

from methods.nerp.network import (
    NeRP2D,
)


def normalize_prior_image(
    prior,
    quantile=0.99,
):
    """
    Normalize a prior magnitude image into the
    approximate [0, 1] intensity convention used
    by NeRP image regression.

    This is preprocessing of the PRIOR ONLY.

    For future longitudinal data the registration /
    resampling of the prior should happen before
    calling this function.
    """

    if torch.is_complex(prior):
        prior = torch.abs(prior)

    prior = prior.float()

    if prior.ndim != 2:
        raise ValueError(
            f"Expected 2-D prior, got {prior.shape}"
        )

    finite = torch.isfinite(prior)

    if not finite.all():

        prior = torch.where(
            finite,
            prior,
            torch.zeros_like(prior),
        )

    positive = prior[
        prior > 0
    ]

    if positive.numel() == 0:
        raise ValueError(
            "Prior contains no positive signal."
        )

    scale = torch.quantile(
        positive,
        quantile,
    )

    scale = torch.clamp(
        scale,
        min=1e-8,
    )

    normalized = (
        prior / scale
    )

    normalized = torch.clamp(
        normalized,
        0.0,
        1.0,
    )

    return (
        normalized,
        scale,
    )


def build_nerp_model(
    device,
    seed=0,
    embedding_size=256,
    fourier_scale=3.0,
    network_depth=8,
    network_width=512,
    w0=30.0,
):
    """
    Construct the MRI configuration of NeRP.
    """

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    encoder = GaussianFourierEncoder(
        coordinates_size=2,
        embedding_size=
            embedding_size,
        scale=
            fourier_scale,
        seed=
            seed,
    )

    model = NeRP2D(
        encoder=encoder,
        depth=
            network_depth,
        width=
            network_width,
        output_size=1,
        w0=
            w0,
    ).to(device)

    return model


def embed_prior(
    prior,
    model=None,
    num_iters=1000,
    learning_rate=1e-4,
    weight_decay=1e-4,
    seed=0,
    embedding_size=256,
    fourier_scale=3.0,
    network_depth=8,
    network_width=512,
    w0=30.0,
    log_every=100,
    verbose=True,
):
    """
    Embed a prior image into NeRP network weights.

    This reproduces the first stage of NeRP:

        theta_prior =
            argmin_theta
            || M_theta(c) - x_prior ||_2^2

    The returned model should then be used as the
    initialization for reconstruction from current
    sparse measurements.
    """

    if not torch.is_tensor(prior):
        prior = torch.as_tensor(
            prior,
            dtype=torch.float32,
        )

    device = prior.device

    (
        normalized_prior,
        prior_scale,
    ) = normalize_prior_image(
        prior
    )

    height, width = (
        normalized_prior.shape
    )

    if model is None:

        model = build_nerp_model(
            device=device,
            seed=seed,
            embedding_size=
                embedding_size,
            fourier_scale=
                fourier_scale,
            network_depth=
                network_depth,
            network_width=
                network_width,
            w0=
                w0,
        )

    else:

        model = model.to(
            device
        )

    grid = make_coordinate_grid(
        height=height,
        width=width,
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )

    history = []

    start_time = time.time()

    for iteration in range(
        num_iters
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = (
            model(grid)[..., 0]
        )

        #
        # Exact source convention:
        #
        #     train_loss =
        #         0.5 * MSE(...)
        #
        loss = (
            0.5
            * F.mse_loss(
                prediction,
                normalized_prior,
            )
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                f"Non-finite prior loss at "
                f"iteration {iteration + 1}"
            )

        loss.backward()

        optimizer.step()

        value = float(
            loss.detach().item()
        )

        history.append(
            value
        )

        if (
            verbose
            and (
                iteration == 0
                or
                (iteration + 1)
                % log_every
                == 0
                or
                iteration
                == num_iters - 1
            )
        ):

            print(
                f"[Prior "
                f"{iteration + 1:4d}/"
                f"{num_iters}] "
                f"loss="
                f"{value:.8e}"
            )

    runtime = (
        time.time()
        - start_time
    )

    model.eval()

    with torch.no_grad():

        embedded_image = (
            model(grid)[..., 0]
        )

        final_mse = (
            F.mse_loss(
                embedded_image,
                normalized_prior,
            )
        ).item()

    return {
        "model":
            model,

        "normalized_prior":
            normalized_prior.detach(),

        "embedded_image":
            embedded_image.detach(),

        "prior_scale":
            float(
                prior_scale.item()
            ),

        "history":
            history,

        "final_mse":
            float(
                final_mse
            ),

        "runtime_sec":
            float(
                runtime
            ),

        "num_iters":
            int(
                num_iters
            ),

        "learning_rate":
            float(
                learning_rate
            ),
    }
