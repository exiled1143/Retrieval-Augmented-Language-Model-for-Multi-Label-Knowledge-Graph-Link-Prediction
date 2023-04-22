from typing import Optional
import torch


class MultiLabelLoss:
    """For multilabel situation.(support focal loss)"""

    def __init__(self, pos_ratio:Optional[float]=1.0, neg_ratio:Optional[float]=1.0, gamma:Optional[float]=0., add_precision_in_loss: str="none"):
        """
        Args:
            pos_ratio: Optional[float]
            neg_ratio: Optional[float]
            gamma: Optional[float]
            add_precision_in_loss: str
                multiply: loss*(1/precision)
                none: loss
        """

        self._pos_ratio = pos_ratio or 1.0
        self._neg_ratio = neg_ratio or 1.0
        self._gamma     = gamma or 0. 
        self._use_mode  = add_precision_in_loss
        self._count = 0
    
    def __repr__(self):
        print(f"MultiLabelLoss(pos_ratio={self._pos_ratio}, neg_ratio={self._neg_ratio}, gamma={self._gamma})")

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        focal loss: alpha1*y*(1-p)^gamma*logp + alpha2*(1-y)*(p)^gamma*log(1-p)
        if use_mode is none:
            loss = focal loss
        if use_mode is multiply:
            loss = focal loss * (1/max(0.01, precision))
        Args:
            inputs: 
                shape: (N, class_num)
            targets: 
                shape: (N, class_num) multi-label onehot 
        Returns:
            loss 
        """

        self._count += 1
        input_sigmoid = inputs.sigmoid() #sigmoid

        pos_loss = -self._pos_ratio * targets * ((1-input_sigmoid)**self._gamma) * torch.log(torch.clip(input_sigmoid, min=1e-9, max=1.))
        neg_loss = -self._neg_ratio * (1.0-targets) * ((input_sigmoid)**self._gamma) *torch.log(torch.clip(1.0-input_sigmoid,min=1e-9, max=1.))
        loss = torch.mean(pos_loss + neg_loss)
        tp = torch.sum(targets[input_sigmoid > 0.5])
        fp = torch.sum(input_sigmoid > 0.5) - tp
        precision = tp/(tp+fp+1e-9)
        precision = max(0.01, precision)
        if self._use_mode == "none":
            return loss
        elif self._use_mode == "multiply":
            return loss * (1./precision)
