from typing import Optional, Union

import torch
from torch import Tensor
from torch.distributions import Distribution,constraints

from sbi.sbi_types import Array
from sbi.utils import BoxUniform

class PostTimesBoxUniform(Distribution):
    def __init__(
        self,
        post,
        post_low: Union[Tensor, Array],
        post_high: Union[Tensor, Array],
        low: Union[Tensor, Array],
        high: Union[Tensor, Array],
        device: Optional[str] = None,
        validate_args=None
    ):
        # Type checks.
        assert isinstance(low, Tensor) and isinstance(high, Tensor) \
            and isinstance(post_low, Tensor) and isinstance(post_high, Tensor), (
            f"bounds must be tensors but are {type(low)}, {type(high)}, {type(post_low)}, and {type(post_high)}."
        )
        if not low.device == high.device == post_low.device == post_high.device:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found at least"
                f"two devices: {low.device}, {high.device}, {post_low.device}, and {post_high.device}."
            )

        # Device handling
        device = low.device.type if device is None else device
        self.device = device

        self.post = post
        self.post_low = torch.as_tensor(post_low, dtype=torch.float32, device=device)
        self.post_high = torch.as_tensor(post_high, dtype=torch.float32, device=device)
        self.post_dim = len(self.post_low)
        self.low = torch.as_tensor(low, dtype=torch.float32, device=device)
        self.high = torch.as_tensor(high, dtype=torch.float32, device=device)
        self.box_uniform = BoxUniform(low,high)
        
        super().__init__(self.box_uniform.batch_shape,
                         torch.Size((self.box_uniform.event_shape[0]+self.post_dim,)),
                         validate_args=validate_args)
        
    @property
    def support(self):
        return constraints._IndependentConstraint(
            constraints._Interval(torch.cat((self.post_low,self.low)),
                                 torch.cat((self.post_high,self.high))),
            1
        )
        
    def sample(self, sample_shape=torch.Size()) -> Tensor:
        post_samples = self.post.sample(sample_shape)
        box_uniform_samples = self.box_uniform.sample(sample_shape)
        if len(box_uniform_samples.shape) == 1:
            post_samples = post_samples[0]
        return torch.cat((post_samples,box_uniform_samples),dim=-1)
    
    def log_prob(self, value):
        return self.post.log_prob(value[:,:self.post_dim]) * self.box_uniform.log_prob(value[:,self.post_dim:])
    
class PostAsPrior(Distribution):
    def __init__(
        self,
        post,
        post_low: Union[Tensor, Array],
        post_high: Union[Tensor, Array],
        device: Optional[str] = None,
        validate_args=None
    ):
        # Type checks.
        assert isinstance(post_low, Tensor) and isinstance(post_high, Tensor), (
            f"bounds must be tensors but are {type(post_low)} and {type(post_high)}."
        )
        if not post_low.device == post_high.device:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found at least"
                f"two devices: {post_low.device} and {post_high.device}."
            )
        
        # Device handling
        device = post_low.device.type if device is None else device
        self.device = device

        self.post = post
        self.post_low = torch.as_tensor(post_low, dtype=torch.float32, device=device)
        self.post_high = torch.as_tensor(post_high, dtype=torch.float32, device=device)
        self.post_dim = len(self.post_low)
        
        super().__init__(torch.Size([]), torch.Size((self.post_dim,)), validate_args=validate_args)
        
    @property
    def support(self):
        return constraints._IndependentConstraint(
            constraints._Interval(self.post_low,self.post_high),
            1
        )
        
    def sample(self, sample_shape=torch.Size()) -> Tensor:
        post_samples = self.post.sample(sample_shape)
        if len(sample_shape) == 0:
            post_samples = post_samples[0]
        return post_samples
    
    def log_prob(self, value):
        return self.post.log_prob(value)
    
class MixturePrior(Distribution):
    def __init__(self, priors, weights=None, validate_args=None):
        self.priors = priors
        self.weights = weights if weights is not None else torch.ones(len(priors)) / len(priors)
        
        # Type checks.
        assert isinstance(weights, Tensor), (
            f"weights must be tensor but is {type(weights)}."
        )
        
        self.low = self.priors[0].support.base_constraint.lower_bound
        self.high = self.priors[0].support.base_constraint.upper_bound
        for prior in self.priors[1:]:
            self.low = torch.min(self.low, prior.support.base_constraint.lower_bound)
            self.high = torch.max(self.high, prior.support.base_constraint.upper_bound)
            
        super().__init__(torch.Size([]), priors[0].event_shape, validate_args=validate_args)

    @property
    def support(self):
        return constraints._IndependentConstraint(
            constraints._Interval(self.low,self.high),
            1
        )

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        if len(sample_shape) == 0: # single sample
            which_to_sample = torch.multinomial(self.weights, num_samples=1, replacement=True)
            return self.priors[which_to_sample].sample(sample_shape)
        # multiple samples
        which_to_sample = torch.multinomial(self.weights, num_samples=sample_shape[0], replacement=True)
        samples = [prior.sample(sample_shape) for prior in self.priors]
        out = samples[0]
        for i in range(1, len(samples)):
            out = torch.where((which_to_sample == i)[:,None], samples[i], out)
        return out

    def log_prob(self, value):
        log_probs = [prior.log_prob(value) for prior in self.priors]
        out = torch.log(self.weights[0]) * log_probs[0]
        for i in range(1, len(log_probs)):
            out += torch.log(self.weights[i]) * log_probs[i]
        return out