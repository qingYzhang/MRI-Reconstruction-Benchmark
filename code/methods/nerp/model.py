import torch


from methods.nerp.prior_embedding import (
    embed_prior,
)

from methods.nerp.reconstruction import (
    reconstruct_nerp_no_prior,
)


def reconstruct_nerp(
    measured_kspace,
    sens_maps,
    mask,
    prior=None,
    prior_iters=1000,
    reconstruction_iters=1000,
    prior_learning_rate=1e-4,
    reconstruction_learning_rate=1e-5,
    embedding_size=256,
    fourier_scale=3.0,
    network_depth=8,
    network_width=512,
    w0=30.0,
    seed=0,
    verbose=True,
):
    """
    Unified NeRP interface.

    prior=None
    ----------
    Randomly initialized INR followed by
    measurement-domain reconstruction.

    prior=image
    -----------
    1. Embed prior image into INR parameters.
    2. Use prior-embedded parameters as the
       initialization for current measurement
       reconstruction.

    IMPORTANT
    ---------
    A supplied prior is assumed to have already
    been registered/resampled into the same
    spatial coordinate system and FOV as the
    current reconstruction.

    This function never requires a ground-truth
    target image.
    """

    device = (
        measured_kspace.device
    )

    prior_result = None
    initial_model = None

    if prior is not None:

        if not torch.is_tensor(
            prior
        ):
            prior = torch.as_tensor(
                prior,
                dtype=torch.float32,
                device=device,
            )

        else:

            prior = prior.to(
                device=device
            )

        prior_result = embed_prior(
            prior=prior,
            model=None,
            num_iters=
                prior_iters,
            learning_rate=
                prior_learning_rate,
            weight_decay=
                1e-4,
            seed=
                seed,
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
            log_every=
                max(
                    1,
                    prior_iters // 10,
                ),
            verbose=
                verbose,
        )

        initial_model = (
            prior_result[
                "model"
            ]
        )

    reconstruction_result = (
        reconstruct_nerp_no_prior(
            measured_kspace=
                measured_kspace,

            sens_maps=
                sens_maps,

            mask=
                mask,

            initial_model=
                initial_model,

            num_iters=
                reconstruction_iters,

            learning_rate=
                reconstruction_learning_rate,

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

            seed=
                seed,

            log_every=
                max(
                    1,
                    reconstruction_iters // 10,
                ),

            verbose=
                verbose,
        )
    )

    reconstruction_result[
        "mode"
    ] = (
        "prior"
        if prior is not None
        else "no_prior"
    )

    if prior_result is None:

        reconstruction_result[
            "prior_embedding"
        ] = None

    else:

        reconstruction_result[
            "prior_embedding"
        ] = {
            "initial_loss":
                prior_result[
                    "history"
                ][0],

            "final_loss":
                prior_result[
                    "history"
                ][-1],

            "final_mse":
                prior_result[
                    "final_mse"
                ],

            "runtime_sec":
                prior_result[
                    "runtime_sec"
                ],

            "num_iters":
                prior_result[
                    "num_iters"
                ],

            "learning_rate":
                prior_result[
                    "learning_rate"
                ],
        }

    return reconstruction_result
