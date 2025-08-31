import torch
import copy
from collections import OrderedDict


def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict

def resume_and_load_fnd(checkpoint, args):
    from collections import OrderedDict
    _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
    ignorelist = []

    def check_keep(keyname, ignorekeywordlist):
        for keyword in ignorekeywordlist:
            if keyword in keyname:
                ignorelist.append(keyname)
                return False
        return True

    _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

    # _load_output = model.load_state_dict(_tmp_st, strict=False)
    # print(_load_output)
    return _tmp_st

def resume_and_load(model, ckpt_path, device, args=None):
    print("Loading checkpoints from", ckpt_path)
    checkpoints = torch.load(ckpt_path, map_location=device)
    if args is not None and args.detector == 'fnd':
        try:
            checkpoints = resume_and_load_fnd(checkpoints['model'], args)
        except KeyError as ke:
            print(f"!!!!!!!!!!!!! Could not find key 'model' in checkpoint")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoints, strict=False)
    elif 'model' in checkpoints.keys() and 'optimizer' in checkpoints.keys():
        checkpoints = convert_official_ckpt(checkpoints, model.state_dict())
        missing_keys, unexpected_keys = model.load_state_dict(checkpoints)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    return model


def save_ckpt(model, save_path, distributed=False):
    print("Saving checkpoints to", save_path)
    state_dict = model.state_dict() if not distributed else model.module.state_dict()
    torch.save(state_dict, save_path)


def selective_reinitialize(model, reinit_ckpt, keep_modules):
    # print("Doing selective reinitialization. Parameters of the model will be reinitialized EXCEPT FOR:")
    for key in copy.deepcopy(list(reinit_ckpt.keys())):
        to_be_reinit = True
        for keep_module in keep_modules:
            if keep_module in key:
                to_be_reinit = False
                break
        if not to_be_reinit:
            reinit_ckpt.pop(key)
            # print(key)
    model.load_state_dict(reinit_ckpt, strict=False)
    return model


def convert_official_ckpt(checkpoints, state_dict):
    checkpoints = checkpoints['model']
    official_keys, new_keys = sorted(list(checkpoints.keys())), sorted(list(state_dict.keys()))
    new_state_dict = {}
    for k_official, k_new in zip(official_keys, new_keys):
        if not k_official.startswith('class'):
            new_state_dict[k_new] = checkpoints[k_official]
        else:
            print("Skipping", k_official)
            new_state_dict[k_new] = state_dict[k_new]
    return new_state_dict