import torch

from .lr_scheduler import WarmupMultiStepLR


def build_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr_mul = 1.0
        if "backbone" in key:
            lr_mul = cfg.MODEL.BACKBONE.LR_MUL
        if "head" in key:
            lr_mul = cfg.MODEL.HEAD.LR_MUL
        params += [{"params": [value], "lr_mul": lr_mul}]
    optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params,
                                                                lr=cfg.SOLVER.BASE_LR,
                                                                weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    return optimizer


def build_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
    
    #return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=cfg.SOLVER.GAMMA)    
