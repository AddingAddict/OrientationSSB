from typing import Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution

from sbi.sbi_types import Array
from sbi.utils import BoxUniform

class PostTimesBoxUniform(Distribution):
    def __init__(
        self,
        post,
        post_dim: int,
        low: Union[Tensor, Array],
        high: Union[Tensor, Array],
        device: Optional[str] = None,
        validate_args=None
    ):

        # Type checks.
        assert isinstance(low, Tensor) and isinstance(high, Tensor), (
            f"low and high must be tensors but are {type(low)} and {type(high)}."
        )
        if not low.device == high.device:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found at least"
                f"two devices, {low.device} and {high.device}."
            )

        # Device handling
        device = low.device.type if device is None else device
        self.device = device

        self.post = post
        self.post_dim = post_dim
        self.low = torch.as_tensor(low, dtype=torch.float32, device=device)
        self.high = torch.as_tensor(high, dtype=torch.float32, device=device)
        self.box_uniform = BoxUniform(low,high)
        
        super().__init__(self.box_uniform.batch_shape,
                         torch.Size((self.box_uniform.event_shape[0]+self.post_dim,)),
                         validate_args=validate_args)
        
    def sample(self, sample_shape=torch.Size()) -> Tensor:
        post_samples = self.post.sample(sample_shape)
        box_uniform_samples = self.box_uniform.sample(sample_shape)
        if len(box_uniform_samples.shape) == 1:
            post_samples = post_samples[0]
        return torch.cat((post_samples,box_uniform_samples),dim=-1)
    
    def log_prob(self, value):
        return self.post.log_prob(value[:,:self.post_dim]) * self.box_uniform.log_prob(value[:,self.post_dim:])