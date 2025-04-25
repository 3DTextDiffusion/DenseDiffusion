import torch
import numpy as np
import diffusers
import pickle

from PIL import Image
from diffusers import DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from transformers.models.clip import CLIPTextModel
from transformers.models.clip.tokenization_clip import CLIPTokenizer
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from torch import Tensor, nn

device = "cpu"
pipe: StableDiffusionPipeline = diffusers.StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    safety_checker=None,
    variant="fp16",
    cache_dir="./models/diffusers/",
).to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(50)

"""
[981, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741, 721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461, 441, 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181, 161, 141, 121, 101, 81, 61, 41, 21, 1]
"""
timesteps = pipe.scheduler.timesteps
sample_size = pipe.unet.sample_size
batch_size = 4

pipe_scheduler: DDIMScheduler = pipe.scheduler
image_processor: VaeImageProcessor = pipe.image_processor
text_encoder: CLIPTextModel = pipe.text_encoder
tokenizer: CLIPTokenizer = pipe.tokenizer
unet: UNet2DConditionModel = pipe.unet


def count_parameters(model: nn.Module, only_trainable=False):
    """
    Print the number of parameters in a PyTorch module.

    Args:
        model (nn.Module): The model to inspect.
        only_trainable (bool): Whether to count only trainable parameters.
    """
    if only_trainable:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {params:,}")
    else:
        params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {params:,}")


print("text_encoder")
count_parameters(text_encoder)
print("unet")
count_parameters(unet)

print()


def mod_forward(
    self,
    hidden_states: Tensor,
    encoder_hidden_states=None,
    attention_mask=None,
    temb=None,
):

    residual = hidden_states

    if self.spatial_norm is not None:
        hidden_states = self.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(
            batch_size, channel, height * width
        ).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )
    attention_mask = self.prepare_attention_mask(
        attention_mask, sequence_length, batch_size
    )

    if self.group_norm is not None:
        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = self.to_q(hidden_states)

    sa_ = True if encoder_hidden_states is None else False
    encoder_hidden_states = (
        text_cond if encoder_hidden_states is not None else hidden_states
    )
    if self.norm_cross:
        encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

    key = self.to_k(encoder_hidden_states)
    value = self.to_v(encoder_hidden_states)

    query = self.head_to_batch_dim(query)
    key = self.head_to_batch_dim(key)
    value = self.head_to_batch_dim(value)

    #################################################
    global COUNT

    if COUNT / 32 < 50 * reg_part:

        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        sim = torch.baddbmm(
            torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device,
            ),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        treg = torch.pow(timesteps[COUNT // 32] / 1000, 5)

        ## reg at self-attn
        if sa_:
            min_value = sim[int(sim.size(0) / 2) :].min(-1)[0].unsqueeze(-1)
            max_value = sim[int(sim.size(0) / 2) :].max(-1)[0].unsqueeze(-1)
            mask = spatial_regularization_maps[sim.size(1)].repeat(self.heads, 1, 1)
            size_reg = regularization_sizes[sim.size(1)].repeat(self.heads, 1, 1)

            sim[int(sim.size(0) / 2) :] += (
                (mask > 0)
                * size_reg
                * sreg
                * treg
                * (max_value - sim[int(sim.size(0) / 2) :])
            )
            sim[int(sim.size(0) / 2) :] -= (
                ~(mask > 0)
                * size_reg
                * sreg
                * treg
                * (sim[int(sim.size(0) / 2) :] - min_value)
            )

        ## reg at cross-attn
        else:
            min_value = sim[int(sim.size(0) / 2) :].min(-1)[0].unsqueeze(-1)
            max_value = sim[int(sim.size(0) / 2) :].max(-1)[0].unsqueeze(-1)
            mask = cross_attention_regularization_maps[sim.size(1)].repeat(
                self.heads, 1, 1
            )
            size_reg = regularization_sizes[sim.size(1)].repeat(self.heads, 1, 1)

            sim[int(sim.size(0) / 2) :] += (
                (mask > 0)
                * size_reg
                * creg
                * treg
                * (max_value - sim[int(sim.size(0) / 2) :])
            )
            sim[int(sim.size(0) / 2) :] -= (
                ~(mask > 0)
                * size_reg
                * creg
                * treg
                * (sim[int(sim.size(0) / 2) :] - min_value)
            )

        attention_probs = sim.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)

    else:
        attention_probs = self.get_attention_scores(query, key, attention_mask)

    COUNT += 1
    #################################################

    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = self.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(
            batch_size, channel, height, width
        )

    if self.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / self.rescale_output_factor

    return hidden_states


for _module in pipe.unet.modules():
    if _module.__class__.__name__ == "Attention":
        _module.__class__.__call__ = mod_forward


with open("./dataset/valset.pkl", "rb") as f:
    dataset = pickle.load(f)
layout_img_root = "./dataset/valset_layout/"

# with open('./dataset/testset.pkl', 'rb') as f:
#     dataset = pickle.load(f)
# layout_img_root = './dataset/testset_layout/'

idx = 5
layout_img_path = layout_img_root + str(idx) + ".png"

"""
[
    'A painting of a couple holding a yellow umbrella in a street on a rainy night. The woman is wearing a white dress and the man is wearing a blue suit.', 
    'a street on a rainy night', 
    'the man is wearing a blue suit',
    'a yellow umbrella',
    'the woman is wearing a white dress'
]
"""
prompts = [dataset[idx]["textual_condition"]] + dataset[idx]["segment_descriptions"]

############
text_input = tokenizer(
    prompts,
    padding="max_length",
    return_length=True,
    return_overflowing_tokens=False,
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
cond_embeddings = text_encoder(text_input.input_ids.to(device))[0]

uncond_input = tokenizer(
    [""] * batch_size,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

for i in range(1, len(prompts)):
    wlen = text_input["length"][i] - 2
    widx = text_input["input_ids"][i][1 : 1 + wlen]
    for j in range(77):
        if (text_input["input_ids"][0][j : j + wlen] == widx).sum() == wlen:
            print("Found", i)
            break

############
# sample_size = 3
# layout_img_ is of shape (512, 512, 3)
layout_img_ = np.asarray(
    Image.open(layout_img_path).resize([sample_size * 8, sample_size * 8])
)[:, :, :3]
unique, counts = np.unique(np.reshape(layout_img_, (-1, 3)), axis=0, return_counts=True)
sorted_idx = np.argsort(-counts)

layouts_ = []

for i in range(len(prompts) - 1):
    if (unique[sorted_idx[i]] == [0, 0, 0]).all() or (
        unique[sorted_idx[i]] == [255, 255, 255]
    ).all():
        layouts_ = [
            ((layout_img_ == unique[sorted_idx[i]]).sum(-1) == 3).astype(np.uint8)
        ] + layouts_
    else:
        layouts_.append(
            ((layout_img_ == unique[sorted_idx[i]]).sum(-1) == 3).astype(np.uint8)
        )
for i, l in enumerate(layouts_):
    Image.fromarray(layouts_[i] * 255, mode="L").save(f"data/layouts/{i}.png")

layouts = [torch.FloatTensor(l).unsqueeze(0).unsqueeze(0).to(device) for l in layouts_]
layouts = F.interpolate(torch.cat(layouts), (sample_size, sample_size), mode="nearest")


############
print("\n".join(prompts))
Image.fromarray(
    np.concatenate([255 * _.squeeze().cpu().numpy() for _ in layouts], 1).astype(
        np.uint8
    )
).save("data/layouts/joint.png")

###########################
###### Preparation for Spatial Regularization maps ######
###########################
spatial_regularization_maps = {}
regularization_sizes = {}
for r in range(4):
    res = int(sample_size / np.power(2, r))
    layouts_s: Tensor = F.interpolate(layouts, (res, res), mode="nearest")
    layouts_s = (
        (
            layouts_s.view(layouts_s.size(0), 1, -1)
            * layouts_s.view(layouts_s.size(0), -1, 1)
        )
        .sum(0)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
    )
    regularization_sizes[np.power(res, 2)] = 1 - 1.0 * layouts_s.sum(
        -1, keepdim=True
    ) / (np.power(res, 2))
    spatial_regularization_maps[np.power(res, 2)] = layouts_s


###########################
###### Preparation for cross-attention regularization ######
###########################
per_word_wise_maps = torch.zeros(1, 77, sample_size, sample_size).to(device)
for i in range(1, len(prompts)):
    wlen = text_input["length"][i] - 2
    widx = text_input["input_ids"][i][1 : 1 + wlen]
    for j in range(77):
        if (text_input["input_ids"][0][j : j + wlen] == widx).sum() == wlen:
            per_word_wise_maps[:, j : j + wlen, :, :] = layouts[i - 1 : i]
            cond_embeddings[0][j : j + wlen] = cond_embeddings[i][1 : 1 + wlen]
            print(prompts[i], i, "-th segment is handled.")
            break

cross_attention_regularization_maps = {}
for r in range(4):
    res = int(sample_size / np.power(2, r))
    layout_c = (
        F.interpolate(per_word_wise_maps, (res, res), mode="nearest")
        .view(1, 77, -1)
        .permute(0, 2, 1)
        .repeat(batch_size, 1, 1)
    )
    cross_attention_regularization_maps[np.power(res, 2)] = layout_c


###########################
#### prep for text_emb ####
###########################
text_cond = torch.cat([uncond_embeddings, cond_embeddings[:1].repeat(batch_size, 1, 1)])

reg_part = 0.3
sreg = 0.3
creg = 1.0

COUNT = 0

with torch.no_grad():
    #     latents = torch.randn(bsz,4,sp_sz,sp_sz).to(device)
    latents = torch.randn(
        batch_size,
        4,
        sample_size,
        sample_size,
        generator=torch.Generator().manual_seed(1),
    ).to(device)
    image = pipe(prompts[:1] * batch_size, latents=latents).images

Image.fromarray(
    np.concatenate(
        [layout_img_.astype(np.uint8)]
        + [np.asarray(image[i]) for i in range(len(image))],
        1,
    )
)
