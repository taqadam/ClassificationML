import argparse

codebook = {
    0: "Cat",
    1: "No Cat"
}


def get_gpus():
    from keras import backend as K
    gpus = K.tensorflow_backend._get_available_gpus()
    return gpus


def get_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yml")
    return parser.parse_args()
