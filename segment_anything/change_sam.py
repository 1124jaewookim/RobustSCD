import numpy as np
import torch
from segment_anything.modeling import Sam
from typing import Optional, Tuple
from torch import nn

class ChangeSam(nn.Module):
    def __init__(
        self,
        sam_model: Sam,
    ) -> None:

        super().__init__()
        self.model = sam_model
        self.input_size = (1024, 1024)
        self.original_size = (1024, 1024)
    def forward(self, imgs):
        img_t0, img_t1  = torch.split(imgs, 3, 1)

        print(img_t0.shape)
        img_embeddings, _ = self.model.image_encoder(img_t0) # (b, 256, 64, 64)
        
        
        fmaps, change_prob_region = self.model.prompt_generator(imgs)
        # print(f"prompt : {fmaps.shape}")

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=None, boxes=None, masks=fmaps)
        # print(f"sparse_embedding : {sparse_embeddings.shape}")
        # print(f"dense_embedding : {dense_embeddings.shape}")
        
        low_res_masks, _ = self.model.mask_decoder(
            image_embeddings=img_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        return masks, change_prob_region

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None