from methods.lacs.official_fista import (
    official_fista_reconstruct,
)

from methods.lacs.fixed_mask import (
    lacs_fixed_mask,
)


def reconstruct_lacs(
    measured_kspace,
    sens_maps,
    mask,
    prior=None,
    wavelet_levels=2,
    outer_rounds=3,
    epsilon1=0.1,
    verbose=False,
):
    """
    Unified reconstruction-side LACS interface.

    prior=None:
        no-prior Wavelab/FISTA reconstruction.

    prior=image:
        prior-informed adaptive LACS.

    The original adaptive acquisition strategy is
    intentionally separated so all reconstruction
    methods can share the same fixed sampling masks.
    """

    if prior is None:

        reconstruction, history = (
            official_fista_reconstruct(
                measured_kspace=measured_kspace,
                sens_maps=sens_maps,
                mask=mask,
                x_prior=None,
                w1=None,
                w2=None,
                wavelet_levels=wavelet_levels,
                num_iters=20,
                L=100.0,
                use_hard_dc=True,
                verbose=verbose,
            )
        )

        return {
            "reconstruction": reconstruction,
            "mode": "no_prior",
            "history": history,
            "rounds": None,
        }

    result = lacs_fixed_mask(
        measured_kspace=measured_kspace,
        sens_maps=sens_maps,
        mask=mask,
        x_prior=prior,
        wavelet_levels=wavelet_levels,
        outer_rounds=outer_rounds,
        epsilon1=epsilon1,
        weight_mode="adaptive",
        verbose=verbose,
    )

    return {
        "reconstruction":
            result["reconstruction"],
        "mode":
            "prior",
        "history":
            None,
        "rounds":
            result["rounds"],
        "scale":
            result["scale"],
    }