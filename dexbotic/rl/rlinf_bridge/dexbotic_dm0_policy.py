# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# DM0 RL policy: merged-attention path (`DM0ForCausalLM`) with the same worker
# interfaces as the Pi0 policy. Loaded via ``ModelRegistry`` from Dexbotic.

import glob
import json
import math
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from dexbotic.data.dataset.transform.action import ActionNorm, PadState
from dexbotic.data.dataset.transform.common import Pipeline, ToNumpy, ToTensor
from dexbotic.data.dataset.transform.output import AbsoluteAction, ActionDenorm
from dexbotic.model.dm0.dm0_arch import DM0Config, DM0ForCausalLM
from dexbotic.model.dm0.dm0_utils import (
    make_attn_mask_2d,
    make_attn_mask_4d,
    make_suffix_attn_mask_2d,
)
from dexbotic.tokenization.process import Pi0Tokenization
from omegaconf import DictConfig
from PIL import Image
from transformers import AutoTokenizer, DynamicCache

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.utils.logging import get_logger

from dexbotic.rl.rlinf_bridge.norm_stats_utils import proprio_pad_ndim_from_norm_stats


class DexboticDM0ForRLActionPrediction(BasePolicy, DM0ForCausalLM):
    """RL policy wrapper for DM0: same *interfaces* as the Pi0 dexbotic policy.

    Differences from Pi0 implementation:
    - Prefix/suffix embeddings: `get_prefix_hidden_states` / `get_suffix_hidden_states`.
    - Attention: `_merged_attention_forward` + `make_*_attn_mask_*` (dm0_utils).
    - Suffix path does not consume proprio/state (DM0 arch); `state` is kept in the
      signature only for API compatibility with the Pi0 policy call sites.
    """

    def __init__(self, config):
        DM0ForCausalLM.__init__(self, config)
        self.logger = get_logger()

        model_dtype = None
        if (
            hasattr(self.model, "llm")
            and hasattr(self.model.llm, "layers")
            and len(self.model.llm.layers) > 0
        ):
            for param in self.model.llm.layers[0].parameters():
                model_dtype = param.dtype
                break
        elif hasattr(self.model, "action_expert") and hasattr(
            self.model.action_expert, "model"
        ):
            layers = self.model.action_expert.model.layers
            if len(layers) > 0:
                for param in layers[0].parameters():
                    model_dtype = param.dtype
                    break
        if model_dtype is None:
            params = list(self.model.parameters())
            model_dtype = params[0].dtype if params else torch.float32
        self.model = self.model.to(dtype=model_dtype)

        if hasattr(self.model, "action_expert") and hasattr(
            self.model.action_expert, "model"
        ):
            self.model.action_expert.model.embed_tokens = None

        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

        self.config = config
        self.num_steps = getattr(config, "num_steps", 10)
        chunk = getattr(config, "chunk_size", None)
        if chunk is None:
            chunk = self.model.config.chunk_size
        self.action_horizon = chunk
        self.num_action_chunks = getattr(config, "output_action_chunks", chunk)
        self.action_dim = config.action_dim
        self.non_delta_mask = getattr(config, "non_delta_mask", [6])
        self.global_step = 0
        self.use_vlm_value = False

        action_hidden = config.action_config.hidden_size
        self.value_head = nn.Linear(action_hidden, 1).to(
            dtype=self.model.action_out_proj.weight.dtype
        )

        self._input_transform = None
        self._output_transform = None
        self.norm_stats = None
        self.pi0_tokenization = None

    # ------------------------------------------------------------------ #
    # VLM freeze (same idea as Pi0 policy)
    # ------------------------------------------------------------------ #
    def freeze_vlm(self):
        if not getattr(self.config, "train_expert_only", False):
            self.logger.warning("freeze_vlm() called but train_expert_only is False")
            return
        if getattr(self.model, "mm_vision_tower", None) is not None:
            self.model.mm_vision_tower.eval()
            for p in self.model.mm_vision_tower.parameters():
                p.requires_grad = False
        if getattr(self.model, "llm", None) is not None:
            self.model.llm.eval()
            for p in self.model.llm.parameters():
                p.requires_grad = False
        if getattr(self.model, "mm_projector", None) is not None:
            self.model.mm_projector.eval()
            for p in self.model.mm_projector.parameters():
                p.requires_grad = False

    def _read_normalization_stats(self, norm_stats_file):
        if not os.path.exists(norm_stats_file):
            raise FileNotFoundError(
                f"Normalization stats not found at {norm_stats_file}. "
                "Ensure norm_stats.json exists next to the checkpoint."
            )
        with open(norm_stats_file, "r") as f:
            norm_stats = json.load(f)
            if "norm_stats" in norm_stats:
                norm_stats = norm_stats["norm_stats"]
        return ToNumpy()(norm_stats)

    def setup_wrappers(self, transforms=(), output_transforms=()):
        self._input_transform = Pipeline(transforms) if transforms else None
        self._output_transform = (
            Pipeline(output_transforms) if output_transforms else None
        )

    def input_transform(self, obs: dict, transpose=True):
        if "prompt" in obs:
            prompts = obs["prompt"]
            if isinstance(prompts, str):
                prompts = [prompts]
            elif isinstance(prompts, torch.Tensor):
                prompts = [str(p) for p in prompts]
            batch_input_ids = []
            for prompt in prompts:
                tokenized = self.pi0_tokenization([{"value": prompt}])
                batch_input_ids.append(tokenized["input_ids"])
            batch_input_ids = torch.from_numpy(np.array(batch_input_ids))
            batch_attention_mask = batch_input_ids != self.tokenizer.pad_token_id
            obs["tokenized_prompt"] = batch_input_ids
            obs["tokenized_prompt_mask"] = batch_attention_mask

        if self._input_transform is not None and "observation/state" in obs:
            state_tensor = obs["observation/state"]
            state_value = (
                state_tensor.cpu().float().numpy()
                if isinstance(state_tensor, torch.Tensor)
                else state_tensor
            )
            state_dict = self._input_transform({"state": state_value})
            obs["observation/state"] = state_dict["state"]
            obs["states"] = state_dict["state"]
        return obs

    def output_transform(self, outputs):
        if self._output_transform is None:
            self.logger.warning(
                "[output_transform] _output_transform is None; actions are not denormalized."
            )
            return outputs

        state_batch = outputs.get("state", None)
        meta_data = outputs.get("meta_data", {})
        batch_size = outputs["actions"].shape[0]
        transformed_actions = []
        for i in range(batch_size):
            sample = {"action": outputs["actions"][i].cpu().numpy()}
            if state_batch is not None:
                sample["state"] = (
                    state_batch[i].cpu().numpy()
                    if isinstance(state_batch, torch.Tensor)
                    else state_batch[i]
                )
            if meta_data:
                sample["meta_data"] = meta_data
            sample = self._output_transform(sample)
            transformed_actions.append(torch.from_numpy(sample["action"]))
        outputs["actions"] = torch.stack(transformed_actions, dim=0).to(
            outputs["actions"].device
        )
        outputs["actions"] = outputs["actions"][:, : self.num_action_chunks]
        return outputs

    def precision_processor(self, processed_obs):
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item)
                    else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if torch.is_tensor(sub_value):
                        processed_obs[key][sub_key] = sub_value.to(
                            device=device
                        ).contiguous()
        return processed_obs

    def forward(self, forward_type="default_forward", **kwargs):
        if "forward_inputs" in kwargs and "data" not in kwargs:
            kwargs["data"] = kwargs.pop("forward_inputs")
        if forward_type == "default_forward":
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"Forward type {forward_type} not implemented")

    def default_forward(self, data, **kwargs):
        compute_values = kwargs.get("compute_values", False)
        chains = data["chains"]
        denoise_inds = data["denoise_inds"]
        observation = (
            data if "tokenized_prompt" in data else self.input_transform(data, transpose=False)
        )

        device = chains.device
        raw_main = observation["observation/image"]
        raw_wrist = observation.get("observation/wrist_image", None)
        images, img_masks = self._process_images_for_training(
            raw_main, raw_wrist, device
        )

        target_dtype = next(self.parameters()).dtype
        lang_tokens = observation["tokenized_prompt"].to(device)
        lang_masks = observation["tokenized_prompt_mask"].to(device)
        state = observation["observation/state"].to(device=device)
        chains = chains.to(device=device, dtype=target_dtype)

        log_probs, value_t, entropy = self.get_log_prob_value(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            chains,
            denoise_inds,
            compute_values,
        )

        action_env_dim = getattr(self.config, "action_env_dim", self.config.action_dim)
        log_probs = log_probs[:, :, : self.num_action_chunks, :action_env_dim]
        entropy = entropy[:, :, : self.num_action_chunks, :action_env_dim]
        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[:, None]
        value_t = value_t.mean(dim=-1, keepdim=False)

        return {"logprobs": log_probs, "values": value_t, "entropy": entropy}

    def _process_images_for_training(self, raw_main_images, raw_wrist_images, device):
        if torch.is_tensor(raw_main_images):
            raw_main_images = raw_main_images.cpu().numpy()
        if raw_wrist_images is not None and torch.is_tensor(raw_wrist_images):
            raw_wrist_images = raw_wrist_images.cpu().numpy()

        batch_size = raw_main_images.shape[0]
        base_pil = []
        for batch_idx in range(batch_size):
            img_np = raw_main_images[batch_idx]
            if img_np.dtype != np.uint8:
                img_np = (
                    (img_np * 255).astype(np.uint8)
                    if img_np.max() <= 1.0
                    else img_np.astype(np.uint8)
                )
            base_pil.append(Image.fromarray(img_np))
        wrist_pil = []
        if raw_wrist_images is not None:
            for batch_idx in range(batch_size):
                wrist_pil.append(
                    Image.fromarray(raw_wrist_images[batch_idx].astype(np.uint8))
                )
        images_list = []
        for batch_idx in range(batch_size):
            if wrist_pil:
                processed = self.process_images(
                    [base_pil[batch_idx], wrist_pil[batch_idx]]
                )
            else:
                processed = self.process_images([base_pil[batch_idx]])
            images_list.append(processed)
        images = torch.stack(images_list, dim=0).to(
            device=device, dtype=next(self.parameters()).dtype
        )

        num_views = images.shape[1]
        required_num_images = 3
        if num_views < required_num_images:
            pad_size = required_num_images - num_views
            padding = torch.zeros(
                batch_size,
                pad_size,
                *images.shape[2:],
                dtype=images.dtype,
                device=device,
            )
            images = torch.cat([images, padding], dim=1)
        image_masks = torch.zeros(
            batch_size, required_num_images, dtype=torch.bool, device=device
        )
        image_masks[:, :num_views] = True
        return images, image_masks

    def _normalize_state(self, state):
        if not hasattr(self, "norm_stats") or self.norm_stats is None:
            return state
        if "state" not in self.norm_stats:
            return state
        stats = self.norm_stats["state"]
        mean = torch.tensor(stats["mean"], device=state.device, dtype=state.dtype)
        std = torch.tensor(stats["std"], device=state.device, dtype=state.dtype)
        return (state - mean) / (std + 1e-6)

    def obs_processor(self, env_obs):
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        state = env_obs["states"]
        if torch.is_tensor(state):
            state = state.to(dtype=torch.float32)
        processed_obs["observation/state"] = state
        if "wrist_images" in env_obs:
            processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
        return processed_obs

    # ------------------------------------------------------------------ #
    # DM0-specific: prefix KV + suffix hidden states (training / logprob)
    # ------------------------------------------------------------------ #
    def _build_prefix_kv_cache(
        self,
        images: torch.Tensor,
        img_masks: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ):
        prefix_hidden, prefix_pad, prefix_attn = self.get_prefix_hidden_states(
            lang_tokens, lang_masks, images, img_masks
        )
        if self.model.config.bf16:
            prefix_hidden = prefix_hidden.to(dtype=torch.bfloat16)

        prefix_2d = make_attn_mask_2d(
            padding_mask=prefix_pad, attn_mask=prefix_attn
        )
        prefix_4d = make_attn_mask_4d(prefix_2d, dtype=prefix_hidden.dtype)
        positions = torch.cumsum(prefix_pad, dim=1) - 1
        module_list = [self.model.llm, self.model.action_expert.model]

        decoder_list, kv_cache = self._merged_attention_forward(
            module_list=module_list,
            attention_mask=prefix_4d,
            position_ids=positions,
            past_key_values=DynamicCache(),
            input_embeds_list=[prefix_hidden, None],
            use_cache=True,
        )
        prefix_output = decoder_list[0]
        return prefix_pad, prefix_attn, kv_cache, prefix_output

    def get_suffix_out(
        self,
        _state,
        prefix_pad: torch.Tensor,
        prefix_attn: torch.Tensor,
        past_key_values: DynamicCache,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """One suffix forward (same mask/positions as DM0 `_denoise_step`), last chunk tokens."""
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=x_t.device)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(x_t.shape[0])

        suffix_hidden, suffix_pad, suffix_attn = self.get_suffix_hidden_states(
            x_t, timestep
        )
        if self.model.config.bf16:
            suffix_hidden = suffix_hidden.to(dtype=torch.bfloat16)

        suffix_2d = make_suffix_attn_mask_2d(
            suffix_padding_mask=suffix_pad,
            suffix_attn_mask=suffix_attn,
            prefix_padding_mask=prefix_pad,
            prefix_attn_mask=prefix_attn,
        )
        suffix_4d = make_attn_mask_4d(suffix_2d, dtype=suffix_hidden.dtype)
        prefix_offsets = torch.sum(prefix_pad, dim=-1)[:, None]
        full_positions = prefix_offsets + torch.cumsum(suffix_pad, dim=1) - 1
        module_list = [self.model.llm, self.model.action_expert.model]

        (_, suffix_out), _ = self._merged_attention_forward(
            module_list=module_list,
            attention_mask=suffix_4d,
            position_ids=full_positions,
            past_key_values=past_key_values,
            input_embeds_list=[None, suffix_hidden],
            use_cache=False,
        )
        chunk = self.model.config.chunk_size
        return suffix_out.clone()[:, -chunk:].to(dtype=next(self.parameters()).dtype)

    @torch.no_grad()
    def sample_actions(
        self, processed_obs, noise=None, mode="train", compute_values=True
    ):
        original_training_mode = self.training
        self.eval()
        try:
            input_ids = processed_obs.get("tokenized_prompt")
            attention_mask = processed_obs.get("tokenized_prompt_mask")
            states = processed_obs["observation/state"].to(
                device=next(self.parameters()).device
            )
            raw_images = processed_obs["observation/image"]
            batch_size = raw_images.shape[0]
            device = states.device

            base_pil = []
            for batch_idx in range(batch_size):
                img_np = raw_images[batch_idx].cpu().numpy()
                if img_np.dtype != np.uint8:
                    img_np = (
                        (img_np * 255).astype(np.uint8)
                        if img_np.max() <= 1.0
                        else img_np.astype(np.uint8)
                    )
                base_pil.append(Image.fromarray(img_np))
            wrist_pil = []
            if "observation/wrist_image" in processed_obs:
                wrist_raw = processed_obs["observation/wrist_image"]
                for batch_idx in range(batch_size):
                    wrist_pil.append(
                        Image.fromarray(wrist_raw[batch_idx].cpu().numpy().astype(np.uint8))
                    )
            images_list = []
            for batch_idx in range(batch_size):
                if wrist_pil:
                    processed = self.process_images(
                        [base_pil[batch_idx], wrist_pil[batch_idx]]
                    )
                else:
                    processed = self.process_images([base_pil[batch_idx]])
                images_list.append(processed)
            images = torch.stack(images_list, dim=0).to(
                device=device, dtype=next(self.parameters()).dtype
            )
            num_views = images.shape[1]
            required_num_images = 3
            if num_views < required_num_images:
                pad_size = required_num_images - num_views
                padding = torch.zeros(
                    batch_size,
                    pad_size,
                    *images.shape[2:],
                    dtype=images.dtype,
                    device=device,
                )
                images = torch.cat([images, padding], dim=1)
            image_masks = torch.zeros(
                batch_size, required_num_images, dtype=torch.bool, device=device
            )
            image_masks[:, :num_views] = True

            model_dtype = next(self.parameters()).dtype
            device_type = next(self.parameters()).device.type
            if model_dtype == torch.bfloat16:
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    actions = self.inference_action(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        states=states,
                        images=images,
                        image_masks=image_masks,
                        diffusion_steps=self.num_steps,
                    )
            else:
                actions = self.inference_action(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    states=states,
                    images=images,
                    image_masks=image_masks,
                    diffusion_steps=self.num_steps,
                )
            dummy_chains = (
                actions.unsqueeze(1)
                .expand(batch_size, self.num_steps + 1, -1, -1)
                .contiguous()
            )
            action_env_dim = getattr(self.config, "action_env_dim", self.config.action_dim)
            return {
                "actions": actions,
                "chains": dummy_chains,
                "prev_logprobs": torch.zeros(
                    batch_size, 10, action_env_dim, device=device
                ),
                "prev_values": torch.zeros(batch_size, 1, device=device),
                "denoise_inds": torch.zeros(
                    batch_size, self.num_steps, dtype=torch.long, device=device
                ),
            }
        finally:
            if original_training_mode:
                self.train()

    def get_logprob_norm(self, sample, mu, sigma):
        if getattr(self.config, "safe_get_logprob", False):
            return -torch.pow((sample - mu), 2)
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
            2 * math.pi * torch.ones_like(sample)
        )
        exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        log_prob = constant_term + exponent_term
        return torch.where(mask, torch.zeros_like(log_prob), log_prob)

    def gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        return 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))

    def get_value_from_vlm(self, prefix_output):
        raise NotImplementedError(
            "use_vlm_value=True is not implemented for DM0 RL policy; "
            "keep use_vlm_value=False or add a pooled prefix head."
        )

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        prefix_pad,
        prefix_attn,
        past_key_values,
        mode,
        denoise_steps,
        compute_values=True,
    ):
        bsize = state.shape[0]
        device = state.device
        if isinstance(idx, int):
            idx = torch.tensor(idx, device=device).expand(bsize)
        if getattr(self.config, "noise_anneal", False):
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            noise_level = torch.tensor(
                getattr(self.config, "noise_level", 0.5)
            ).to(device)

        denoise_steps = int(denoise_steps)
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]

        suffix_out = self.get_suffix_out(
            state, prefix_pad, prefix_attn, past_key_values, x_t, t_input
        )
        w_dtype = self.model.action_out_proj.weight.dtype
        v_t = self.model.action_out_proj(suffix_out.to(dtype=w_dtype))

        if (
            getattr(self.config, "add_value_head", False)
            and compute_values
            and not getattr(self.config, "value_after_vlm", False)
        ):
            chunk_sz = self.model.config.chunk_size
            if getattr(self.config, "chunk_critic_input", True):
                suffix_out_value = torch.mean(
                    suffix_out[:, :chunk_sz], dim=1, keepdim=False
                )
            else:
                suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
            if getattr(self.config, "detach_critic_input", False):
                suffix_out_value = suffix_out_value.detach()
            value_t = self.value_head(
                suffix_out_value.to(self.value_head.weight.dtype)
            )[:, 0]
        else:
            value_t = torch.zeros((bsize), device=device)

        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            noise_method = getattr(self.config, "noise_method", "flow_sde")
            if noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        timesteps
                        / (
                            1
                            - torch.where(
                                timesteps == 1, timesteps[1], timesteps
                            )
                        )
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i
            elif noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = (t_input - delta) * cos_term
                x_t_std = (t_input - delta) * sin_term
            elif noise_method == "flow_noise":
                if not hasattr(self, "noise_head"):
                    raise NotImplementedError(
                        "noise_method='flow_noise' requires self.noise_head "
                        "(see Pi0 RL policy); not wired for DM0."
                    )
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.noise_head(
                    suffix_out.to(dtype=self.model.action_out_proj.weight.dtype)
                )
            else:
                raise ValueError(f"Invalid noise method: {noise_method}")
        else:
            raise ValueError(mode)

        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t

    def get_log_prob_value(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        chains,
        denoise_inds,
        compute_values=False,
    ):
        bsize = state.shape[0]
        prefix_pad, prefix_attn, past_key_values, prefix_output = (
            self._build_prefix_kv_cache(
                images, img_masks, lang_tokens, lang_masks
            )
        )

        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        if getattr(self.config, "joint_logprob", False):
            num_steps = self.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1

        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind].clone()
            chains_next = chains[torch.arange(bsize), denoise_ind + 1].clone()
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                state,
                prefix_pad,
                prefix_attn,
                past_key_values,
                "train",
                self.num_steps,
                compute_values,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)

            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)

            if self.use_vlm_value:
                chains_values.append(self.get_value_from_vlm(prefix_output))
            else:
                chains_values.append(value_t)

        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)
        if getattr(self.config, "noise_method", "") == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)

        return chains_log_probs, chains_values, chains_entropy

    def predict_action_batch(self, env_obs, **kwargs):
        mode = kwargs.get("mode", "train")
        compute_values = kwargs.get("compute_values", True)
        to_process_obs = self.obs_processor(env_obs)
        processed_obs = self.input_transform(to_process_obs, transpose=False)
        processed_obs = self.precision_processor(processed_obs)

        outputs = self.sample_actions(
            processed_obs=processed_obs, mode=mode, compute_values=compute_values
        )
        if hasattr(self, "_output_transform") and self._output_transform is not None:
            state_for_transform = processed_obs.get("observation/state")
            if state_for_transform is not None:
                state_numpy = (
                    state_for_transform.cpu().numpy()
                    if isinstance(state_for_transform, torch.Tensor)
                    else state_for_transform
                )
                meta_data = {"non_delta_mask": np.array(self.non_delta_mask)}
                outputs["state"] = state_numpy
                outputs["meta_data"] = meta_data
                outputs = self.output_transform(outputs)
            else:
                outputs = self.output_transform(outputs)

        action_env_dim = getattr(self.config, "action_env_dim", self.config.action_dim)
        actions = outputs["actions"][:, :, :action_env_dim]
        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
        }
        if "tokenized_prompt" in processed_obs:
            forward_inputs["tokenized_prompt"] = processed_obs["tokenized_prompt"]
        if "tokenized_prompt_mask" in processed_obs:
            forward_inputs["tokenized_prompt_mask"] = processed_obs[
                "tokenized_prompt_mask"
            ]
        forward_inputs.update(to_process_obs)
        forward_inputs.pop("prompt", None)
        return actions, {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }


def get_model(cfg: DictConfig, torch_dtype: Optional[Any] = None):
    import safetensors.torch

    logger = get_logger()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    if not cfg.model_path or not os.path.exists(cfg.model_path):
        raise ValueError(f"Model path does not exist: {cfg.model_path}")

    try:
        config = DM0Config.from_pretrained(cfg.model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path, use_fast=False, local_files_only=True
        )

        config.num_steps = cfg.get("num_steps", getattr(config, "num_steps", 10))
        config.action_env_dim = cfg.action_dim
        config.output_action_chunks = cfg.num_action_chunks
        config.add_value_head = cfg.get("add_value_head", True)
        config.noise_level = cfg.get("dexbotic", {}).get("noise_level", 0.5)
        config.noise_method = cfg.get("dexbotic", {}).get("noise_method", "flow_sde")
        config.detach_critic_input = cfg.get("dexbotic", {}).get(
            "detach_critic_input", True
        )
        config.train_expert_only = cfg.get("dexbotic", {}).get(
            "train_expert_only", False
        )
        config.safe_get_logprob = cfg.get("safe_get_logprob", False)
        config.chunk_critic_input = cfg.get("chunk_critic_input", True)
        config.noise_anneal = cfg.get("noise_anneal", False)
        config.joint_logprob = cfg.get("joint_logprob", False)
        config.value_after_vlm = cfg.get("value_after_vlm", False)
        config.processor_config = cfg.model_path

        original_offline = os.environ.get("HF_HUB_OFFLINE", None)
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            model = DexboticDM0ForRLActionPrediction(config)
        finally:
            if original_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = original_offline

        model.tokenizer = tokenizer
        model.pi0_tokenization = Pi0Tokenization(tokenizer)

        weight_paths = sorted(glob.glob(os.path.join(cfg.model_path, "*.safetensors")))
        weight_paths = [p for p in weight_paths if not p.endswith(".index.json")]
        if not weight_paths:
            weight_path = os.path.join(cfg.model_path, "model.safetensors")
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"No weights found in {cfg.model_path}")
            weight_paths = [weight_path]
        for weight_path in weight_paths:
            state_dict = safetensors.torch.load_file(weight_path)
            model_keys = {n for n, _ in model.named_parameters()}
            state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
            model.load_state_dict(state_dict, strict=False)

        norm_stats_file = os.path.join(cfg.model_path, "norm_stats.json")
        if os.path.exists(norm_stats_file):
            model.norm_stats = model._read_normalization_stats(norm_stats_file)
        else:
            model.norm_stats = None

        model._train_expert_only = getattr(config, "train_expert_only", False)

    except Exception as e:
        logger.error(f"Failed to load pretrained DM0 model: {e}")
        raise

    input_transforms_list = []
    if model.norm_stats is not None:
        state_pad_ndim = proprio_pad_ndim_from_norm_stats(model.norm_stats)
        input_transforms_list = [
            PadState(ndim=state_pad_ndim, axis=-1),
            ActionNorm(statistic_mapping=model.norm_stats, strict=False),
            ToTensor(),
        ]
    output_transforms_list = []
    if model.norm_stats is not None:
        output_transforms_list = [
            ToNumpy(),
            ActionDenorm(statistic_mapping=model.norm_stats, strict=False),
            AbsoluteAction(),
        ]
    model.setup_wrappers(
        transforms=input_transforms_list, output_transforms=output_transforms_list
    )
    return model
