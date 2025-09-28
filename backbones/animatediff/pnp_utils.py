import torch.nn.functional as F
import torch
from einops import rearrange


# Modified from tokenflow_utils.py
def register_time(model, t):
    up_res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # breakpoint()
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, "idx", t)
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, "idx", t)


def register_spatial_attention_pnp(model, eta1=0.0, eta2=0.5):
    def ca_forward(self):
        def forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            clip_length: int = None,
            SparseCausalAttention_index: list = [-1, 'first']
        ) -> torch.FloatTensor:
            input_ndim = hidden_states.ndim
            # breakpoint()
            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            chunk_size = batch_size // 3

            query = self.to_q(hidden_states)
            dim = query.shape[-1]
            head_dim = dim // self.heads
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)
            # breakpoint()
            # -------------------------------------AdaIN-Guided Attention-shift-----------------------------
            if self.idx >= self.eta1 * 50 and self.idx < self.eta2 * 50:
                # parameter
                alpha = 0.8
                # beta = (0.1 - 0.9) / (self.eta2 * 50 - self.eta1 * 50) * (self.idx - self.eta1 * 50) + 0.9
                beta = (1.0 - 0.0) / (self.eta1 * 50 - self.eta2 * 50) * (self.idx - self.eta2 * 50) + 0.0
                # beta = 0
                gamma = 2.0
                # max_config:
                query[2 * chunk_size: 3 * chunk_size] = alpha * query[: chunk_size] + (1 - alpha) * query[2 * chunk_size: 3 * chunk_size]  
                key[2 * chunk_size: 3 * chunk_size] = beta * attention_adain(key[2 * chunk_size: 3 * chunk_size], key[chunk_size: 2 * chunk_size]) + (1 - beta) * key[chunk_size: 2 * chunk_size]
                value[2 * chunk_size: 3 * chunk_size] = beta * attention_adain(value[2 * chunk_size: 3 * chunk_size], value[chunk_size: 2 * chunk_size]) + (1 - beta) * value[chunk_size: 2 * chunk_size]
                # attn argue
                query[2 * chunk_size: 3 * chunk_size] = gamma * query[2 * chunk_size: 3 * chunk_size]
            # -----------------------------------cross-frame attention--------------------------------------
            if clip_length is not None:
                key = rearrange(key, "(b f) d c -> b f d c", f=clip_length)
                value = rearrange(value, "(b f) d c -> b f d c", f=clip_length)
                #  *********************** Start of Spatial-temporal attention **********
                frame_index_list = []
                if len(SparseCausalAttention_index) > 0:
                    for index in SparseCausalAttention_index:
                        if isinstance(index, str):
                            if index == 'first':
                                frame_index = [0] * clip_length
                            if index == 'last':
                                frame_index = [clip_length - 1] * clip_length
                            if (index == 'mid') or (index == 'middle'):
                                frame_index = [int(clip_length-1)//2] * clip_length
                        else:
                            assert isinstance(index, int), 'relative index must be int'
                            frame_index = torch.arange(clip_length) + index
                            frame_index = frame_index.clip(0, clip_length-1)        
                        frame_index_list.append(frame_index) 
                    key = torch.cat([   key[:, frame_index] for frame_index in frame_index_list
                                        ], dim=2)
                    value = torch.cat([ value[:, frame_index] for frame_index in frame_index_list
                                        ], dim=2)
                #  *********************** End of Spatial-temporal attention **********
                key = rearrange(key, "b f d c -> (b f) d c", f=clip_length)
                value = rearrange(value, "b f d c -> (b f) d c", f=clip_length)
            # ----------------------------------------------------------------------------------------
            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)
            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states
        return forward

    # breakpoint()
    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            # breakpoint()
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, "eta1", eta1)
            setattr(module, "eta2", eta2)
            model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1.forward = ca_forward(module)


def attention_adain(cnt_feat, sty_feat, ad=True):
    beta = 1.0
    cnt_mean = cnt_feat.mean(dim=[1],keepdim=True)
    cnt_std = cnt_feat.std(dim=[1],keepdim=True)
    sty_mean = sty_feat.mean(dim=[1],keepdim=True)
    sty_std = sty_feat.std(dim=[1],keepdim=True)
    output_mean = beta * sty_mean + (1 - beta) * cnt_mean
    output_std = beta * sty_std + (1 - beta) * cnt_std
    if ad:
        output = F.instance_norm(cnt_feat) * output_std + output_mean

    return output.to(cnt_feat.dtype)


def latent_adain(cnt_feat, sty_feat, ad=True):
    beta = 1.0
    cnt_mean = cnt_feat.mean(dim=[0, 3, 4], keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 3, 4], keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 3, 4], keepdim=True)
    sty_std = sty_feat.std(dim=[0, 3, 4], keepdim=True)
    output_mean = beta * sty_mean + (1 - beta) * cnt_mean
    output_std = beta * sty_std + (1 - beta) * cnt_std
    if ad:
        output = F.instance_norm(cnt_feat) * output_std + output_mean

    return output.to(cnt_feat.dtype)