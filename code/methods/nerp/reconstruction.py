import time

import torch


from methods.nerp.encoding import (
    GaussianFourierEncoder,
    make_coordinate_grid,
)

from methods.nerp.network import (
    NeRP2D,
)

from operators.sense import (
    sense_adjoint,
    sense_forward,
)


def estimate_fixed_phase_and_scale(
    measured_kspace,
    sens_maps,
    mask,
    quantile=0.99,
):
    """
    Estimate two nuisance quantities using ONLY
    current undersampled measurements:

        1. a fixed image-domain phase field
        2. a robust image intensity scale

    NeRP itself remains a scalar real-valued
    intensity representation, matching the
    original method.

    No target / fully sampled reconstruction is
    used here.
    """

    with torch.no_grad():

        adjoint = sense_adjoint(
            measured_kspace,
            sens_maps,
            mask,
        )

        magnitude = torch.abs(
            adjoint
        )

        scale = torch.quantile(
            magnitude.reshape(-1),
            quantile,
        )

        scale = torch.clamp(
            scale,
            min=1e-8,
        )

        phase = (
            adjoint
            / torch.clamp(
                magnitude,
                min=1e-8,
            )
        )

        #
        # Phase is meaningless where the signal
        # is essentially zero. Use zero phase there.
        #
        reliable = (
            magnitude
            > 0.01 * scale
        )

        phase = torch.where(
            reliable,
            phase,
            torch.ones_like(
                phase
            ),
        )

    return (
        phase.detach(),
        scale.detach(),
    )


def acquired_complex_mse(
    prediction,
    target,
    mask,
):
    """
    L2 measurement-domain loss normalized by
    the acquired sampling fraction.

    sense_forward already applies the mask, so
    residual is zero at unsampled locations.

    Dividing by sampling fraction makes loss
    magnitude approximately independent of R.
    """

    residual = (
        prediction
        - target
    )

    sampling_fraction = (
        mask.float()
        .mean()
        .clamp_min(
            1e-8
        )
    )

    return (
        torch.mean(
            torch.abs(
                residual
            ) ** 2
        )
        / sampling_fraction
    )


def reconstruct_nerp_no_prior(
    measured_kspace,
    sens_maps,
    mask,
    initial_model=None,
    num_iters=50,
    learning_rate=1e-5,
    embedding_size=256,
    fourier_scale=3.0,
    network_depth=8,
    network_width=512,
    w0=30.0,
    seed=0,
    log_every=10,
    verbose=True,
):
    """
    NeRP without prior embedding.

    This corresponds to the NeRP w/o prior
    ablation: randomly initialized INR optimized
    directly using sparse measurements.

    fastMRI adaptation:
        Original MRI forward model:
            NUFFT(single scalar image)

        Here:
            Cartesian multi-coil SENSE.

    The scalar NeRP representation itself is
    unchanged.
    """

    device = (
        measured_kspace.device
    )

    height = int(
        measured_kspace.shape[-2]
    )

    width = int(
        measured_kspace.shape[-1]
    )

    if initial_model is None:

        torch.manual_seed(
            seed
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed
            )

        encoder = (
            GaussianFourierEncoder(
                coordinates_size=2,
                embedding_size=
                    embedding_size,
                scale=
                    fourier_scale,
                seed=
                    seed,
            )
        )

        model = NeRP2D(
            encoder=encoder,
            depth=
                network_depth,
            width=
                network_width,
            output_size=1,
            w0=w0,
        ).to(device)

        initialization = "random"

    else:

        #
        # NeRP prior embedding:
        #
        # continue optimization from the
        # prior-embedded network parameters.
        #
        # The Fourier matrix B is part of the
        # supplied model and is NOT regenerated.
        #
        model = initial_model.to(
            device
        )

        if (
            model.encoder.B.shape[-1]
            != 2
        ):
            raise ValueError(
                "Prior-embedded NeRP model "
                "does not use 2-D coordinates."
            )

        initialization = (
            "prior_embedded"
        )
    
    grid = make_coordinate_grid(
        height=height,
        width=width,
        device=device,
    )

    (
        fixed_phase,
        measurement_scale,
    ) = estimate_fixed_phase_and_scale(
        measured_kspace=
            measured_kspace,

        sens_maps=
            sens_maps,

        mask=
            mask,
    )

    scaled_measurements = (
        measured_kspace
        / measurement_scale
    )

    #
    # Absorb the measurement-derived common
    # image phase into the SENSE maps.
    #
    effective_sens_maps = (
        sens_maps
        * fixed_phase.unsqueeze(0)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    history = []

    start_time = time.time()

    for iteration in range(
        num_iters
    ):

        optimizer.zero_grad(
            set_to_none=True
        )

        #
        # Scalar real image:
        # [H,W,1] -> [H,W]
        #
        normalized_image = (
            model(grid)[..., 0]
        )

        predicted_kspace = (
            sense_forward(
                normalized_image.to(
                    torch.complex64
                ),
                effective_sens_maps,
                mask,
            )
        )

        loss = acquired_complex_mse(
            prediction=
                predicted_kspace,

            target=
                scaled_measurements,

            mask=
                mask,
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                f"Non-finite loss at "
                f"iteration "
                f"{iteration + 1}"
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
                or (
                    iteration + 1
                ) % log_every
                == 0
                or iteration
                == num_iters - 1
            )
        ):
            print(
                f"[NeRP "
                f"{iteration + 1:4d}/"
                f"{num_iters}] "
                f"kspace_loss="
                f"{value:.8e}"
            )

    with torch.no_grad():

        normalized_image = (
            model(grid)[..., 0]
        )

        complex_reconstruction = (
            normalized_image
            * measurement_scale
            * fixed_phase
        )

    runtime = (
        time.time()
        - start_time
    )

    return {
        "reconstruction":
            complex_reconstruction,

        "normalized_image":
            normalized_image.detach(),

        "model":
            model,

        "history":
            history,

        "measurement_scale":
            float(
                measurement_scale.item()
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

        "mode":
            (
                "no_prior"
                if initial_model is None
                else "prior_initialized"
            ),

        "initialization":
            initialization,
    }
