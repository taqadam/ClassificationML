import os
import numpy as np
import pandas as pd

from pprint import pprint
from model.input_fn import DataLoader
from model.model_fn import Model
from model.trainer import Trainer
from model.utils import get_gpus, get_user_args
from configs import parser


def setup(configs):
    config_name = configs["name"]
    checkpoint_dir = configs["checkpoint_dir"]
    logs_dir = configs["logs_dir"]
    trained_model_dir = configs["trained_model_dir"]
    tensorboard_dir = configs["tensorboard_dir"]
    results_dir = configs["results_dir"]

    notes = [
        checkpoint_dir,
        logs_dir,
        tensorboard_dir,
        trained_model_dir,
        results_dir,
    ]

    for dst_dir in notes:
        os.makedirs(f"{dst_dir}/{config_name}", exist_ok=True)

    configs["callbacks"]["checkpoint"]["filepath"] = f"{checkpoint_dir}/{config_name}/{config_name}.h5"
    configs["callbacks"]["csv_logger"]["filename"] = f"{logs_dir}/{config_name}/{config_name}.csv"
    configs["callbacks"]["tensorboard"]["log_dir"] = f"{tensorboard_dir}/{config_name}"

    trained_model_dir = configs["trained_model_dir"]
    configs["trained_model"]["filepath"] = f"{trained_model_dir}/{config_name}/{config_name}.h5"

    results_dir = configs["results_dir"]
    configs["results_fp"] = f"{results_dir}/{config_name}/{config_name}.csv"

    return configs


def main():
    gpus = get_gpus()
    print("GPUs: {}".format(gpus))

    args = get_user_args()
    configs = parser.parse(args.config)
    params = setup(configs)

    data_loader = DataLoader()
    train_datagen = data_loader.get_datagen('train', params)
    val_datagen = data_loader.get_datagen('val', params)

    model = Model(params)
    model.create_model()
    model.inspect()

    trainer = Trainer(
        model.model,
        train_datagen,
        val_datagen,
        params,
    )

    pprint(params)
    trainer.add_callbacks(params)
    trainer.train(params)
    trainer.model.save(
        **params["trained_model"]
    )

    test_datagen = data_loader.get_datagen('test', params)
    step_size = test_datagen.n // test_datagen.batch_size
    preds = trainer.model.predict_generator(
        test_datagen, verbose=1, steps=step_size
    )
    predicted_class_indices = np.argmax(preds, axis=1)
    labels = dict((v, k) for k, v in (train_datagen.class_indices).items())
    predictions = [labels[k] for k in predicted_class_indices]
    filenames = test_datagen.filenames
    results = pd.DataFrame(
        {
            "file": filenames,
            "prediction": predictions,
            "prediction_class": predicted_class_indices,
            "cat_proba": [p[0] for p in preds],
            "no_cat_proba": [p[1] for p in preds],
            "label": test_datagen.classes,
        }
    )
    results_fp = params["results_fp"]
    results.to_csv(f"{results_fp}", index=False)
    print(results)


if __name__ == '__main__':
    main()
