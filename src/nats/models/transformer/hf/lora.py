from torch import nn
from torchtune.modules.peft.lora import LoRALinear


def get_fine_tune_models(tune_pars):
    if tune_pars == 'all':
        return None
    else:
        involved_modules = []
        for tune_par in tune_pars:
            if tune_par == 'q':
                involved_modules.append('q_proj')
            if tune_par == 'k':
                involved_modules.append('k_proj')
            if tune_par == 'v':
                involved_modules.append('v_proj')
            if tune_par == 'o':
                involved_modules.append('o_proj')
        return involved_modules


def config_lora_for_model(model, lora_config: dict| None = None, involved_modules: set | None = None):
    lora_config = lora_config or {}
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    for n, n_modules in model.named_modules():
        if (n.startswith('layers')) and not n.endswith('proj_layer'):
            if involved_modules is None or n.split('.')[-1] in involved_modules:
                if isinstance(n_modules, nn.Linear):
                    print(n)
                    adapting_layer = LoRALinear(in_dim=n_modules.in_features,
                                                out_dim=n_modules.out_features,
                                                rank=lora_config.get('lora_rank', 8),
                                                alpha=lora_config.get('lora_alpha', 16),
                                                use_bias=n_modules.bias is not None
                                                )
                    adapting_layer = adapting_layer.to(device=device, dtype=dtype)
                    adapting_layer.weight.data = n_modules.weight.data
                    # we freeze the bias weights
                    adapting_layer.weight.requires_grad = False
                    if adapting_layer.bias is not None:
                        adapting_layer.bias.requires_grad = False
                #elif isinstance(n_modules, (LayerNorm, RMSNorm)):
                #    adapting_layer = NormLayerAdapter(base_layer=n_modules)

                else:
                    continue
                module_ = model
                all_atts = n.split('.')
                for att_name in all_atts[:-1]:
                    if att_name.isnumeric():
                        module_ = module_[int(att_name)]
                    else:
                        module_ = getattr(module_, att_name)
                setattr(module_, all_atts[-1], adapting_layer)
