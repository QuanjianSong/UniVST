import torch
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.models.attention import Attention
import torch.nn.functional as F
from einops import rearrange
import random


class CrossFrameProcessor:
    """Attention processor used typically in processing the SD3-like self-attention projections."""
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("JointAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        idx = -1,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        # additional
        clip_length = 16
        SparseCausalAttention_index = ['first', -1, 0] # ['first', -1, 0] better settings

        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads


        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        

        # 在这里加跨帧注意力
        # breakpoint()
        if clip_length is not None:
            # breakpoint()
            # key = rearrange(key, "(b f) d c -> b f d c", f=clip_length)
            # value = rearrange(value, "(b f) d c -> b f d c", f=clip_length)
            key = rearrange(key, '(b f) h n c -> b f h n c', f=clip_length)
            value = rearrange(value, '(b f) h n c -> b f h n c', f=clip_length)
            #  *********************** Start of Spatial-temporal attention **********
            frame_index_list = []
            if len(SparseCausalAttention_index) > 0:
                for index in SparseCausalAttention_index:
                    if isinstance(index, str):
                        if index == 'first':
                            frame_index = [0] * clip_length
                        elif index == 'last':
                            frame_index = [clip_length - 1] * clip_length
                        elif (index == 'mid') or (index == 'middle'):
                            frame_index = [int(clip_length-1)//2] * clip_length
                    else:
                        assert isinstance(index, int), 'relative index must be int'
                        frame_index = torch.arange(clip_length) + index
                        frame_index = frame_index.clip(0, clip_length-1)        
                    frame_index_list.append(frame_index) 
                key = torch.cat([   key[:, frame_index] for frame_index in frame_index_list
                                    ], dim=-2)
                value = torch.cat([ value[:, frame_index] for frame_index in frame_index_list
                                    ], dim=-2)
            # breakpoint()
            #  *********************** End of Spatial-temporal attention **********
            key = rearrange(key, "b f h n c -> (b f) h n c", f=clip_length)
            value = rearrange(value, "b f h n c -> (b f) h n c", f=clip_length)
        


        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)


        


        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class AttentionShiftProcessor:
    """Attention processor used typically in processing the SD3-like self-attention projections."""
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("JointAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        idx = -1, 
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        # additional
        clip_length = 16
        SparseCausalAttention_index = ['first', -1, 0]

        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        
        
        # breakpoint()

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # breakpoint() sd3 这里不会走，sd3.5会走
        if attn.norm_q is not None:
            # breakpoint()
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)


        # 在这里加attention-shift
        # breakpoint()
        chunk_size = batch_size // 3
        # breakpoint()
        self.thresh1 = 0.0
        self.thresh2 = 0.6 # sd-3 0.6， sd-3.5 0.5
        # 感觉0.0～0.6比较合适，0.7有点过了
        # -------------------------------------AdaIN-Guided Attention-shift-----------------------------
        if idx >= self.thresh1 * 50 and idx <= self.thresh2 * 50:
            # parameter
            alpha = 0.8
            # beta = (0.9 - 0.1) / (1000 - 400) * (self.idx - 400) + 0.1 # origin
            beta = (0.9 - 0.1) / (self.thresh1 * 50 - self.thresh2 * 50) * (idx - self.thresh2 * 50) + 0.1 # 单调减
            # beta = (0.0 - 1.0) / (self.thresh1 * 50 - self.thresh2 * 50) * (idx - self.thresh2 * 50) + 1.0

            # beta = 0.0
            gamma = 2.0
            # max_config:
            query[2 * chunk_size: 3 * chunk_size] = alpha * query[: chunk_size] + (1 - alpha) * query[2 * chunk_size: 3 * chunk_size]  
            key[2 * chunk_size: 3 * chunk_size] = beta * attention_adain(key[2 * chunk_size: 3 * chunk_size], key[chunk_size: 2 * chunk_size]) + (1 - beta) * key[chunk_size: 2 * chunk_size]
            value[2 * chunk_size: 3 * chunk_size] = beta * attention_adain(value[2 * chunk_size: 3 * chunk_size], value[chunk_size: 2 * chunk_size]) + (1 - beta) * value[chunk_size: 2 * chunk_size]
            # attn argue
            query[2 * chunk_size: 3 * chunk_size] = gamma * query[2 * chunk_size: 3 * chunk_size]
        # ----------------------------------------------------------------------------------------------


        # 在这里加跨帧注意力
        # breakpoint()
        # if idx >= self.thresh1 * 50 and idx <= self.thresh2 * 50:
        if True:
            if clip_length is not None:
                # breakpoint()
                # key = rearrange(key, "(b f) d c -> b f d c", f=clip_length)
                # value = rearrange(value, "(b f) d c -> b f d c", f=clip_length)
                key = rearrange(key, '(b f) h n c -> b f h n c', f=clip_length)
                value = rearrange(value, '(b f) h n c -> b f h n c', f=clip_length)
                #  *********************** Start of Spatial-temporal attention **********
                frame_index_list = []
                if len(SparseCausalAttention_index) > 0:
                    for index in SparseCausalAttention_index:
                        if isinstance(index, str):
                            if index == 'first':
                                frame_index = [0] * clip_length
                            elif index == 'last':
                                frame_index = [clip_length - 1] * clip_length
                            elif (index == 'mid') or (index == 'middle'):
                                frame_index = [int(clip_length-1)//2] * clip_length
                        else:
                            assert isinstance(index, int), 'relative index must be int'
                            frame_index = torch.arange(clip_length) + index
                            frame_index = frame_index.clip(0, clip_length-1)        
                        frame_index_list.append(frame_index) 
                    key = torch.cat([   key[:, frame_index] for frame_index in frame_index_list
                                        ], dim=-2)
                    value = torch.cat([ value[:, frame_index] for frame_index in frame_index_list
                                        ], dim=-2)
                # breakpoint()
                #  *********************** End of Spatial-temporal attention **********
                key = rearrange(key, "b f h n c -> (b f) h n c", f=clip_length)
                value = rearrange(value, "b f h n c -> (b f) h n c", f=clip_length)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            # breakpoint()
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states




def register_spatial_attention_pnp(model, thresh=0.5):
    new_procs = {}
    # breakpoint()
    for idx, (name, processor) in enumerate(model.transformer.attn_processors.items()):
        # if idx >= 12:
        if idx >= 0:
            if 'attn' in name:
                new_procs[name] = AttentionShiftProcessor()
            else:
                new_procs[name] = processor
        else:
            new_procs[name] = processor
    model.transformer.set_attn_processor(new_procs)


def attention_adain(cnt_feat, sty_feat, ad=True):
    beta = 1.0

    cnt_mean = cnt_feat.mean(dim=[-2],keepdim=True)
    cnt_std = cnt_feat.std(dim=[-2],keepdim=True)
    sty_mean = sty_feat.mean(dim=[-2],keepdim=True)
    sty_std = sty_feat.std(dim=[-2],keepdim=True)

    output_mean = beta * sty_mean + (1 - beta) * cnt_mean
    output_std = beta * sty_std + (1 - beta) * cnt_std
    if ad:
        output = F.instance_norm(cnt_feat) * output_std + output_mean

    return output.to(cnt_feat.dtype)


def latent_adain(cnt_feat, sty_feat, ad=True):
    beta = 1.0
    cnt_mean = cnt_feat.mean(dim=[2, 3], keepdim=True)
    cnt_std = cnt_feat.std(dim=[2, 3], keepdim=True)
    sty_mean = sty_feat.mean(dim=[2, 3], keepdim=True)
    sty_std = sty_feat.std(dim=[2, 3], keepdim=True)
    output_mean = beta * sty_mean + (1 - beta) * cnt_mean
    output_std = beta * sty_std + (1 - beta) * cnt_std
    if ad:
        output = F.instance_norm(cnt_feat) * output_std + output_mean

    return output.to(cnt_feat.dtype)
