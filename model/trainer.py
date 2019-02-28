from pprint import pprint

from keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
)


class Trainer(object):
    def __init__(self, model, train_generator, val_generator, params):
        self.callbacks = []
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.model = model

    def add_callbacks(self, params):
        checkpoint = ModelCheckpoint(
            **params["callbacks"]["checkpoint"]
        )
        self.callbacks.append(checkpoint)

        early_stopping = EarlyStopping(
            **params["callbacks"]["early_stopping"]
        )
        self.callbacks.append(early_stopping)

        tensorboard = TensorBoard(
            **params["callbacks"]["tensorboard"]
        )
        self.callbacks.append(tensorboard)

        csv_logger = CSVLogger(
            **params["callbacks"]["csv_logger"]
        )
        self.callbacks.append(csv_logger)

    def train(self, params):
        batch_size = params["generators"]["train"]["batch_size"]
        params["trainer"]["steps_per_epoch"] = self.train_generator.n // batch_size
        params["trainer"]["validation_steps"] = self.val_generator.n // batch_size
        print("-" * 90)
        pprint(params, indent=4)
        print("-" * 90)
        self.history = self.model.fit_generator(
            self.train_generator,
            validation_data=self.val_generator,
            callbacks=self.callbacks,
            **params["trainer"],
        )
