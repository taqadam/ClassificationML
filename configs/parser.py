import yaml
from pathlib import Path
from pprint import pprint
from keras.optimizers import adam, sgd


def parse(config_fp):
    with open(config_fp) as fp:
        configs = dict(yaml.load(fp))

    BASE_NAME = Path(config_fp).stem

    configs["name"] = BASE_NAME

    # add optimizer here!
    learning_rate = configs["model"]["compilier"]["learning_rate"]
    optimizers = {
        "adam": adam(lr=learning_rate),
        "sgd": sgd(lr=learning_rate),
    }

    configs["model"]["compilier"]["optimizer"] = optimizers[
        configs["model"]["compilier"]["optimizer"]
    ]

    configs["training"]["input_shape"] = tuple(
        configs["training"]["input_shape"]
    )

    configs["model"]["base"]["input_shape"] = tuple(
        configs["model"]["base"]["input_shape"]
    )

    configs["training"]["target_size"] = tuple(
        configs["training"]["target_size"]
    )
    configs["generators"]["train"]["target_size"] = tuple(
        configs["generators"]["train"]["target_size"]
    )
    configs["generators"]["test"]["target_size"] = tuple(
        configs["generators"]["test"]["target_size"]
    )
    configs["generators"]["val"]["target_size"] = tuple(
        configs["generators"]["val"]["target_size"]
    )

    configs["model"]["base"]["input_shape"] = tuple(
        configs["model"]["base"]["input_shape"]
    )

    tensorboard_dir = configs["checkpoint_dir"]
    configs["callbacks"]["tensorboard"]["log_dir"] = f"{tensorboard_dir}/{BASE_NAME}"

    return configs


if __name__ == "__main__":
    params = parse("baseline.yml")
    pprint(params)
