from keras.applications.vgg16 import VGG16 as pre_train_model
from keras.layers import Dense, Flatten
from keras.models import Sequential


class Model(object):

    def __init__(self, params):

        self.input_shape = params["model"]["base"]["input_shape"]
        self.include_top = params["model"]["base"]["include_top"]
        self.weights = params["model"]["base"]["weights"]
        self.block = params["model"]["base"]["block"]

        self.loss = params["model"]["compilier"]["loss"]
        self.optimizer = params["model"]["compilier"]["optimizer"]
        self.learning_rate = params["model"]["compilier"]["learning_rate"]
        self.metrics = params["model"]["compilier"]["metrics"]

    def set_base(self):
        self.base = pre_train_model(
            weights=self.weights,
            include_top=self.include_top,
            input_shape=self.input_shape
        )

        set_trainable = False

        for layer in self.base.layers:
            if layer.name == self.block:
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

            self.base.trainable = False

    def create_model(self):
        self.model = Sequential()
        self.set_base()
        self.model.add(self.base)
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(2, activation='sigmoid'))

        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics,
        )

        return self.model

    def inspect(self):
        self.model.summary()
        for layer in self.base.layers:
            if layer.trainable:
                print("Retraining: {}".format(layer.name))
        return True
