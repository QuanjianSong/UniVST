import os

import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class CustomSD3Transformer2DModel(SD3Transformer2DModel):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
        #
        idx=0,
        ft_indices: str = None,
        ft_timesteps: int = None,
        ft_path: str = None,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states, ip_temb = self.image_proj(ip_adapter_image_embeds, timestep)

            joint_attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)

        for index_block, block in enumerate(self.transformer_blocks):
            # Skip specified layers
            is_skip = True if skip_layers is not None and index_block in skip_layers else False

            if torch.is_grad_enabled() and self.gradient_checkpointing and not is_skip:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    joint_attention_kwargs,
                )
            elif not is_skip:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
            
            # -----------------------------------additional-----------------------------------
            if ft_indices is not None and ft_timesteps and ft_path is not None:
                # save feature map
                if index_block in ft_indices and idx == ft_timesteps:
                    save_path = os.path.join(ft_path, f'inversion_feature_map_{index_block}_block_{idx}_step.pt')
                    print(f'save feature map at: {save_path}')
                    torch.save(hidden_states.view(hidden_states.shape[0], height//2, width//2, -1).detach(),
                               os.path.join(ft_path, save_path))


            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) / len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[int(index_block / interval_control)]

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
