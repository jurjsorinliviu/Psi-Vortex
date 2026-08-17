"""Validated kernel-density BIC and persistent global weight clustering."""
from __future__ import annotations
import math
import copy
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def exact_effective_dof(weights:torch.Tensor,gamma:float,block_size:int=1024)->torch.Tensor:
    """Original kernel expression with block-bounded forward and recomputed backward."""
    flat=weights.flatten(); total=flat.new_zeros(())
    for start in range(0,flat.numel(),block_size):
        def block_dof(block,all_weights):
            density=torch.exp(-((block[:,None]-all_weights[None,:])**2)/(2*gamma*gamma)).sum(1)
            return (1./density.clamp_min(1e-12)).sum()
        block=flat[start:start+block_size]
        # Small models are safer and faster with ordinary autograd; checkpointed
        # recomputation is reserved for vectors where retained pairwise blocks matter.
        use_checkpoint=flat.requires_grad and flat.numel()>4096
        total=total+(checkpoint(block_dof,block,flat,use_reentrant=False) if use_checkpoint else block_dof(block,flat))
    return total


class DifferentiableBIC(nn.Module):
    def __init__(self,gamma:float=.1,eps:float=1e-8,block_size:int=1024): super().__init__(); self.gamma,self.eps,self.block_size=gamma,eps,block_size
    def effective_dof(self,model:nn.Module):
        weights=torch.cat([p.reshape(-1) for p in model.parameters() if p.requires_grad])
        return exact_effective_dof(weights,self.gamma,self.block_size)
    def complexity(self,model:nn.Module,n_observations:int):
        if n_observations<2: raise ValueError("BIC requires at least two scalar target elements")
        return math.log(n_observations)*self.effective_dof(model)
    def forward(self,model:nn.Module,mse:torch.Tensor,n_observations:int):
        return n_observations*torch.log(mse.clamp_min(self.eps))+self.complexity(model,n_observations)


class GloballyClusteredModel(nn.Module):
    """A functional model with one trainable global centroid vector and fixed assignments."""
    def __init__(self,model:nn.Module,n_clusters:int):
        super().__init__(); self.base=model; names=[]; shapes=[]; assignments=[]
        named=list(model.named_parameters()); flat=torch.cat([p.detach().reshape(-1) for _,p in named]); count=min(n_clusters,flat.numel())
        centers=torch.linspace(float(flat.min()),float(flat.max()),count,device=flat.device,dtype=flat.dtype)
        for _ in range(30):
            labels=(flat[:,None]-centers[None,:]).abs().argmin(1); updated=torch.stack([flat[labels==i].mean() if torch.any(labels==i) else centers[i] for i in range(count)])
            if torch.allclose(updated,centers): break
            centers=updated
        self.centroids=nn.Parameter(centers); offset=0
        for index,(name,p) in enumerate(named):
            size=p.numel(); assignment=(flat[offset:offset+size,None]-centers[None,:]).abs().argmin(1).reshape(p.shape)
            self.register_buffer(f"assignment_{index}",assignment); names.append(name); shapes.append(tuple(p.shape)); offset+=size; p.requires_grad_(False)
        self._parameter_names=names; self._parameter_shapes=shapes; self.materialized_cluster_count=count
    def expanded_parameters(self):
        return {name:self.centroids[getattr(self,f"assignment_{i}")] for i,name in enumerate(self._parameter_names)}
    def forward(self,*args,**kwargs):
        return torch.func.functional_call(self.base,self.expanded_parameters(),args,kwargs)
    def step(self,x_t:torch.Tensor,state:torch.Tensor):
        """Streaming inference with the currently tied centroid values."""
        return self.bake().step(x_t,state)
    def bake(self)->nn.Module:
        """Create a conventional module snapshot for deployment."""
        baked=copy.deepcopy(self.base)
        with torch.no_grad():
            for name,parameter in baked.named_parameters(): parameter.copy_(self.expanded_parameters()[name])
        return baked


def materialize_weight_clusters(model:nn.Module,n_clusters:int)->GloballyClusteredModel:
    if n_clusters<1: raise ValueError("n_clusters must be positive")
    return GloballyClusteredModel(model,n_clusters)


def select_cluster_count(model:nn.Module,candidates:list[int],loss_fn,n_observations:int,gamma:float=.1):
    """Select a global cluster count by full BIC on caller-supplied calibration loss."""
    if not candidates: raise ValueError("at least one candidate cluster count is required")
    scored=[]
    for count in candidates:
        clustered=materialize_weight_clusters(model,count); mse=loss_fn(clustered)
        score=float(DifferentiableBIC(gamma)(clustered,mse,n_observations).detach()); scored.append((score,count,clustered))
    return min(scored,key=lambda item:item[0])
