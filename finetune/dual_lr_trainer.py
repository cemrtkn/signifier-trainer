from transformers import Trainer


class DualLRTrainer(Trainer):
    """Stock Trainer with the embedding surface on its own learning rate.

    No create_scheduler override: HF schedulers hold a step->factor multiplier
    that is applied to each param group's own base lr, so the stock scheduler
    already gives both groups a shared warmup/decay curve at their own scales.
    """

    def __init__(self, *args, embedding_lr: float, model_lr: float, **kwargs):
        self.embedding_lr = embedding_lr
        self.model_lr = model_lr
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """One optimizer with the input + output embedding matrices at
        embedding_lr, everything else at model_lr split into decay / no-decay
        like the stock optimizer. Membership is by id(p), so a tied
        input/output tensor is counted once and lands in the embedding group
        only."""
        if self.optimizer is None:
            model = self.model
            modules = [model.get_input_embeddings(), model.get_output_embeddings()]
            embed_ids = {
                id(p) for m in modules if m is not None for p in m.parameters()
            }
            decay = set(self.get_decay_parameter_names(model))
            wd = self.args.weight_decay
            groups = [
                {
                    "params": [p for p in model.parameters() if id(p) in embed_ids],
                    "lr": self.embedding_lr,
                    "weight_decay": wd,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if id(p) not in embed_ids and n in decay
                    ],
                    "lr": self.model_lr,
                    "weight_decay": wd,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if id(p) not in embed_ids and n not in decay
                    ],
                    "lr": self.model_lr,
                    "weight_decay": 0.0,
                },
            ]
            if self.optimizer_cls_and_kwargs is not None:
                opt_cls, opt_kwargs = self.optimizer_cls_and_kwargs
            else:
                opt_cls, opt_kwargs = self.get_optimizer_cls_and_kwargs(
                    self.args, model
                )
            opt_kwargs = {
                k: v
                for k, v in opt_kwargs.items()
                if k not in ("lr", "weight_decay", "params")
            }
            self.optimizer = opt_cls(groups, **opt_kwargs)
        return self.optimizer
