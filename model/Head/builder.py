import tensorflow as  tf

def HeadBuild(config):
    if config["model_config"]["head"]["name"].upper() in ["CENTERNET"]:
        from model.Head.CenterNet import CenterNet
        return CenterNet
    else:
        raise ValueError(config["model_config"]["head"]["name"] + " is not implemented yet or misspelled")