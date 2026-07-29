#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from dataclasses import dataclass, field


@dataclass
class DiffusionConfig:
    """Configuration class for DiffusionPolicy.

    Defaults are configured for training with PushT providing proprioceptive and single camera observations.

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and `output_shapes`.

    Notes on the inputs and outputs:
        - "observation.state" is required as an input key.
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.image" they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        horizon: Diffusion model action prediction size as detailed in `DiffusionPolicy.select_action`.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            See `DiffusionPolicy.select_action` for more details.
        input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
            the input data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
            indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
            include batch dimension or temporal dimension.
        output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
            the output data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
            Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
        input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
            and the value specifies the normalization mode to apply. The two available modes are "mean_std"
            which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
            [-1, 1] range.
        output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
            original scale. Note that this is also used for normalizing the training targets.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        crop_shape: (H, W) shape to crop images to as a preprocessing step for the vision backbone. Must fit
            within the image size. If None, no cropping is done.
        crop_is_random: Whether the crop should be random at training time (it's always a center crop in eval
            mode).
        pretrained_backbone_weights: Pretrained weights from torchvision to initalize the backbone.
            `None` means no pretrained weights.
        use_group_norm: Whether to replace batch normalization with group normalization in the backbone.
            The group sizes are set to be about 16 (to be precise, feature_dim // 16).
        spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax.
        down_dims: Feature dimension for each stage of temporal downsampling in the diffusion modeling Unet.
            You may provide a variable number of dimensions, therefore also controlling the degree of
            downsampling.
        kernel_size: The convolutional kernel size of the diffusion modeling Unet.
        n_groups: Number of groups used in the group norm of the Unet's convolutional blocks.
        diffusion_step_embed_dim: The Unet is conditioned on the diffusion timestep via a small non-linear
            network. This is the output dimension of that network, i.e., the embedding dimension.
        use_film_scale_modulation: FiLM (https://arxiv.org/abs/1709.07871) is used for the Unet conditioning.
            Bias modulation is used be default, while this parameter indicates whether to also use scale
            modulation.
        noise_scheduler_type: Name of the noise scheduler to use. Supported options: ["DDPM", "DDIM"].
        num_train_timesteps: Number of diffusion steps for the forward diffusion schedule.
        beta_schedule: Name of the diffusion beta schedule as per DDPMScheduler from Hugging Face diffusers.
        beta_start: Beta value for the first forward-diffusion step.
        beta_end: Beta value for the last forward-diffusion step.
        prediction_type: The type of prediction that the diffusion modeling Unet makes. Choose from "epsilon"
            or "sample". These have equivalent outcomes from a latent variable modeling perspective, but
            "epsilon" has been shown to work better in many deep neural network settings.
        clip_sample: Whether to clip the sample to [-`clip_sample_range`, +`clip_sample_range`] for each
            denoising step at inference time. WARNING: you will need to make sure your action-space is
            normalized to fit within this range.
        clip_sample_range: The magnitude of the clipping range as described above.
        num_inference_steps: Number of reverse diffusion steps to use at inference time (steps are evenly
            spaced). If not provided, this defaults to be the same as `num_train_timesteps`.
        do_mask_loss_for_padding: Whether to mask the loss when there are copy-padded actions. See
            `LeRobotDataset` and `load_previous_and_future_frames` for mor information. Note, this defaults
            to False as the original Diffusion Policy implementation does the same.
        arm_action_dim: Number of action dimensions modeled by the arm diffusion head. If None, the
            dual-head policy chooses a conservative default from the total action dimension.
        view_action_dim: Number of action dimensions modeled by the view/head diffusion head. If None, it is
            inferred from the total action dimension and `arm_action_dim`.
        view_loss_weight: Loss weight for the view/head action head when using `dual_head_diffusion`.
        coupling_num_heads: Number of attention heads used for bidirectional bottleneck coupling.
        coupling_dropout: Dropout probability applied to coupling attention and residuals.
        coupling_mode: Temporal routing used by coupled dual-head diffusion. ``full`` keeps the
            original all-to-all bottleneck exchange. ``rbac`` couples the executed view prefix with
            the unexecuted arm suffix across the next replanning boundary. ``balanced_lookahead``
            splits the bottleneck tokens into equal temporal halves and uses each head's future half
            to refine only the other head's current half. ``rcla`` applies role-causal local lag
            masks: Arm token i reads View tokens {i-1, i}, while View token i reads Arm tokens
            {i, i+1}. ``bidirectional_prefix_to_suffix`` lets each head's future suffix read only
            the other head's execution-boundary prefix, leaving both prefixes unchanged.
            ``bidirectional_half_prefix_to_suffix`` splits the bottleneck into equal halves, then
            lets suffix tokens in each head read only the other head's prefix half.
        coupling_block_type: Conditioning block used around cross-head attention. ``scalar_gate``
            preserves the original LayerNorm plus scalar timestep gate. ``role_adaln_zero`` adds
            role-specific, timestep-conditioned adaptive LayerNorm with zero initialization.
        coupling_use_temporal_pos_emb: Add a fixed one-dimensional sinusoidal position embedding
            to bottleneck tokens used by cross-head attention.
        coupling_use_ffn: Add a zero-gated, role-specific feed-forward residual after cross-head
            attention. This option requires ``coupling_block_type=role_adaln_zero``.
        coupling_ffn_ratio: Hidden-width multiplier of the optional coupling feed-forward network.
        coupling_active_max_timestep: Inclusive maximum diffusion timestep at which coupling
            residuals are enabled. ``None`` preserves coupling at every denoising step; an integer
            ``T`` applies the hard low-noise mask ``1[t <= T]`` to Attention and optional FFN
            residuals. Diffusion timesteps decrease during inference, so a smaller value activates
            coupling only near the end of denoising.
        view_to_arm_coupling_scale: Ablation multiplier applied to the View-to-Arm coupling
            residual. ``0`` disables only View-to-Arm coupling.
        arm_to_view_coupling_scale: Ablation multiplier applied to the Arm-to-View coupling
            residual. ``0`` disables only Arm-to-View coupling.
        scid_ridge: Ridge regularization used when fitting the fixed Arm-to-View
            linear predictor in normalized action coordinates.
        scid_residual_eps: Minimum per-dimension innovation scale used to avoid
            division by zero for nearly deterministic View joints.
        scid_clamp_reconstructed_view: Clamp reconstructed normalized View actions
            to the diffusion scheduler's configured sample range before unnormalization.
    """

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    input_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "observation.image": [3, 96, 96],
            "observation.state": [2],
        }
    )
    output_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "action": [2],
        }
    )

    # Normalization / Unnormalization
    input_normalization_modes: dict[str, str] = field(
        default_factory=lambda: {
            "observation.image": "mean_std",
            "observation.state": "min_max",
        }
    )
    output_normalization_modes: dict[str, str] = field(default_factory=lambda: {"action": "min_max"})

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    debug_save_input_images: bool = False
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    # Unet.
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    # Noise scheduler.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: int | None = None

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Exponential moving average used by the training entrypoint for evaluation/deployment.
    use_ema: bool = False
    ema_decay: float = 0.999
    ema_update_after_step: int = 1000

    # Dual-head diffusion policy. These fields are ignored by the original single-head diffusion policy.
    arm_action_dim: int | None = None
    view_action_dim: int | None = None
    view_loss_weight: float = 0.2
    coupling_num_heads: int = 8
    coupling_dropout: float = 0.0
    coupling_mode: str = "full"
    coupling_block_type: str = "scalar_gate"
    coupling_use_temporal_pos_emb: bool = False
    coupling_use_ffn: bool = False
    coupling_ffn_ratio: float = 2.0
    coupling_active_max_timestep: int | None = None
    view_to_arm_coupling_scale: float = 1.0
    arm_to_view_coupling_scale: float = 1.0
    scid_ridge: float = 1e-3
    scid_residual_eps: float = 1e-6
    scid_clamp_reconstructed_view: bool = True

    @classmethod
    def from_dict(cls, values: dict) -> "DiffusionConfig":
        """加载配置字典，并忽略旧checkpoint遗留的可配置动作起点。"""
        values = dict(values)
        values.pop("action_start", None)
        return cls(**values)

    def __post_init__(self):
        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        image_keys = {k for k in self.input_shapes if k.startswith("observation.image")}

        if len(image_keys) == 0 and "observation.environment_state" not in self.input_shapes:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if len(image_keys) > 0:
            if self.crop_shape is not None:
                for image_key in image_keys:
                    if (
                        self.crop_shape[0] > self.input_shapes[image_key][1]
                        or self.crop_shape[1] > self.input_shapes[image_key][2]
                    ):
                        raise ValueError(
                            f"`crop_shape` should fit within `input_shapes[{image_key}]`. Got {self.crop_shape} "
                            f"for `crop_shape` and {self.input_shapes[image_key]} for "
                            "`input_shapes[{image_key}]`."
                        )
            # Check that all input images have the same shape.
            first_image_key = next(iter(image_keys))
            for image_key in image_keys:
                if self.input_shapes[image_key] != self.input_shapes[first_image_key]:
                    raise ValueError(
                        f"`input_shapes[{image_key}]` does not match `input_shapes[{first_image_key}]`, but we "
                        "expect all image shapes to match."
                    )

        if self.arm_action_dim is not None and self.arm_action_dim <= 0:
            raise ValueError(f"`arm_action_dim` must be positive or None. Got {self.arm_action_dim}.")
        if self.view_action_dim is not None and self.view_action_dim <= 0:
            raise ValueError(f"`view_action_dim` must be positive or None. Got {self.view_action_dim}.")
        if self.view_loss_weight < 0:
            raise ValueError(f"`view_loss_weight` must be non-negative. Got {self.view_loss_weight}.")
        if not isinstance(self.use_ema, bool):
            raise ValueError(f"`use_ema` must be a bool. Got {self.use_ema!r}.")
        if not math.isfinite(self.ema_decay) or not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(
                f"`ema_decay` must be finite and in [0, 1). Got {self.ema_decay}."
            )
        if (
            isinstance(self.ema_update_after_step, bool)
            or not isinstance(self.ema_update_after_step, int)
            or self.ema_update_after_step < 0
        ):
            raise ValueError(
                "`ema_update_after_step` must be a non-negative integer. "
                f"Got {self.ema_update_after_step!r}."
            )
        if not math.isfinite(self.scid_ridge) or self.scid_ridge < 0:
            raise ValueError(
                f"`scid_ridge` must be finite and non-negative. Got {self.scid_ridge}."
            )
        if not math.isfinite(self.scid_residual_eps) or self.scid_residual_eps <= 0:
            raise ValueError(
                "`scid_residual_eps` must be finite and positive. "
                f"Got {self.scid_residual_eps}."
            )
        supported_coupling_modes = {
            "full",
            "rbac",
            "balanced_lookahead",
            "rcla",
            "bidirectional_prefix_to_suffix",
            "bidirectional_half_prefix_to_suffix",
        }
        if self.coupling_mode not in supported_coupling_modes:
            raise ValueError(
                "`coupling_mode` must be 'full', 'rbac', 'balanced_lookahead', "
                "'rcla', 'bidirectional_prefix_to_suffix', or "
                "'bidirectional_half_prefix_to_suffix'. Got "
                f"{self.coupling_mode!r}."
            )
        supported_coupling_block_types = {"scalar_gate", "role_adaln_zero"}
        if self.coupling_block_type not in supported_coupling_block_types:
            raise ValueError(
                "`coupling_block_type` must be 'scalar_gate' or 'role_adaln_zero'. "
                f"Got {self.coupling_block_type!r}."
            )
        for field_name in (
            "coupling_use_temporal_pos_emb",
            "coupling_use_ffn",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"`{field_name}` must be a bool.")
        if self.coupling_use_ffn and self.coupling_block_type != "role_adaln_zero":
            raise ValueError(
                "`coupling_use_ffn=True` requires "
                "`coupling_block_type='role_adaln_zero'`."
            )
        if not math.isfinite(self.coupling_ffn_ratio) or self.coupling_ffn_ratio <= 0:
            raise ValueError(
                "`coupling_ffn_ratio` must be finite and positive. "
                f"Got {self.coupling_ffn_ratio}."
            )
        if self.coupling_active_max_timestep is not None:
            threshold = self.coupling_active_max_timestep
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, int)
                or threshold < 0
                or threshold >= self.num_train_timesteps
            ):
                raise ValueError(
                    "`coupling_active_max_timestep` must be None or an integer in "
                    f"[0, num_train_timesteps). Got "
                    f"{threshold!r} with "
                    f"num_train_timesteps={self.num_train_timesteps}."
                )
        for field_name in (
            "view_to_arm_coupling_scale",
            "arm_to_view_coupling_scale",
        ):
            scale = float(getattr(self, field_name))
            if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
                raise ValueError(
                    f"`{field_name}` must be finite and in [0, 1]. Got {scale}."
                )
        if self.n_obs_steps <= 0:
            raise ValueError(f"`n_obs_steps` must be positive. Got {self.n_obs_steps}.")
        if self.n_action_steps <= 0:
            raise ValueError(f"`n_action_steps` must be positive. Got {self.n_action_steps}.")
        action_start = self.n_obs_steps - 1
        action_end = action_start + self.n_action_steps
        if action_end > self.horizon:
            raise ValueError(
                "The original Diffusion Policy action slice must fit inside the prediction horizon. Got "
                f"start={action_start}, steps={self.n_action_steps}, horizon={self.horizon}."
            )
        if self.coupling_mode in {
            "rbac",
            "bidirectional_prefix_to_suffix",
        } and action_end >= self.horizon:
            raise ValueError(
                "The selected prefix-to-suffix coupling mode requires an unexecuted action "
                "suffix after the replanning boundary. Got "
                f"start={action_start}, steps={self.n_action_steps}, horizon={self.horizon}."
            )
        if self.arm_action_dim is not None and self.view_action_dim is not None:
            action_dim = self.output_shapes["action"][0]
            if self.arm_action_dim + self.view_action_dim != action_dim:
                raise ValueError(
                    f"Dual-head action dims must sum to output action dim. Got "
                    f"arm={self.arm_action_dim}, view={self.view_action_dim}, action={action_dim}."
                )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )

        # Check that the horizon size and U-Net downsampling is compatible.
        # U-Net downsamples by 2 with each stage.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )
