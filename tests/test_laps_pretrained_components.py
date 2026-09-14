import os

import torch

import diffusers

from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_unconditional import (
    StableDiffusionUnconditionalPipeline,
)

from laps.configs.recon import sd_laps_medvae_4
from laps.model.med_vae_wrapper import MedVAEWrapper

from laps.recon_test import Config, load_ds

from laps.dataloaders.labels import (
    PRIOR_KEY,
)


def main():

    print("=" * 80)
    print("LAPS PRETRAINED COMPONENT / REAL-PRIOR VAE TEST")
    print("=" * 80)

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    params = sd_laps_medvae_4
    model_id = params.model_name_or_path

    print("device:", device)
    print("model:", model_id)
    print("diffusers path:", os.path.realpath(diffusers.__file__))

    #
    # ------------------------------------------------------------
    # Load the exact components used by StableDiffusionReconstructor.
    # ------------------------------------------------------------
    #

    print("\nLoading UNet...")

    unet = UNet2DModel.from_pretrained(
        model_id,
        subfolder="unet",
        revision=None,
    )

    unet.eval()

    print("UNet loaded.")

    print("\nLoading MedVAE...")

    vae = MedVAEWrapper.from_pretrained(
        model_id,
        subfolder="vae",
    )

    vae.eval()

    print("MedVAE loaded.")

    #
    # ------------------------------------------------------------
    # Build the author's patched unconditional pipeline.
    # ------------------------------------------------------------
    #

    print("\nLoading pipeline config...")

    pipeline = StableDiffusionUnconditionalPipeline.from_pretrained(
        model_id,
        vae=vae,
        unet=unet,
        revision=None,
        variant=None,
        trust_remote_code=True,
    )

    #
    # Match StableDiffusionReconstructor:
    #
    #   prediction_type = v_prediction
    #   scheduler       = DDIM
    #
    pipeline.scheduler.register_to_config(
        prediction_type=params.prediction_type
    )

    pipeline.scheduler = DDIMScheduler.from_config(
        pipeline.scheduler.config
    )

    print("Pipeline loaded.")

    print("\n--- MODEL CONFIG ---")
    print(
        "UNet class:",
        type(pipeline.unet).__name__,
    )
    print(
        "VAE class:",
        type(pipeline.vae).__name__,
    )
    print(
        "scheduler:",
        type(pipeline.scheduler).__name__,
    )
    print(
        "prediction type:",
        pipeline.scheduler.config.prediction_type,
    )
    print(
        "VAE input channels:",
        pipeline.vae.config.in_channels,
    )
    print(
        "VAE scaling factor:",
        pipeline.vae.config.scaling_factor,
    )
    print(
        "UNet input channels:",
        pipeline.unet.config.in_channels,
    )

    assert isinstance(
        pipeline.scheduler,
        DDIMScheduler,
    )

    assert (
        pipeline.scheduler.config.prediction_type
        == "v_prediction"
    )

    assert pipeline.vae.config.in_channels == 2

    #
    # ------------------------------------------------------------
    # Load the REAL SLAM longitudinal prior.
    # ------------------------------------------------------------
    #

    args = Config(
        dataset="slam-test",
        middle_slices_only=True,
        max_recons=1,
        retrospective_undersampling=False,
    )

    dataset = load_ds(args)

    sample = dataset[0]

    prior = sample[PRIOR_KEY].to(
        device=device,
        dtype=torch.complex64,
    )

    #
    # Match LAPS complex normalization:
    #
    #   x /= max(|x|)
    #   [real, imag]
    #
    prior = prior / prior.abs().max()

    prior_2ch = torch.stack(
        [
            prior.real,
            prior.imag,
        ],
        dim=0,
    ).unsqueeze(0)

    print("\n--- REAL PRIOR ---")
    print(
        "complex prior:",
        tuple(prior.shape),
    )
    print(
        "2-channel prior:",
        tuple(prior_2ch.shape),
    )
    print(
        "prior max abs:",
        prior.abs().max().item(),
    )
    print(
        "prior finite:",
        torch.isfinite(prior_2ch).all().item(),
    )

    #
    # ------------------------------------------------------------
    # VAE encode/decode functionality test.
    # ------------------------------------------------------------
    #

    pipeline.vae.to(device)

    with torch.no_grad():

        encoded = pipeline.vae.encode(
            prior_2ch
        )

        latents = (
            encoded.latent_dist.mode()
            * pipeline.vae.config.scaling_factor
        )

        decoded = pipeline.vae.decode(
            latents
            / pipeline.vae.config.scaling_factor,
            return_dict=False,
        )[0]

    print("\n--- VAE ROUNDTRIP ---")
    print(
        "latent shape:",
        tuple(latents.shape),
    )
    print(
        "decoded shape:",
        tuple(decoded.shape),
    )
    print(
        "latent finite:",
        torch.isfinite(latents).all().item(),
    )
    print(
        "decoded finite:",
        torch.isfinite(decoded).all().item(),
    )

    assert latents.ndim == 4

    assert decoded.shape == prior_2ch.shape

    assert torch.isfinite(
        latents
    ).all()

    assert torch.isfinite(
        decoded
    ).all()

    print("\n" + "=" * 80)
    print("ALL PRETRAINED COMPONENT / VAE CHECKS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
