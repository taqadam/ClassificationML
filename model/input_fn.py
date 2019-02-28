from keras.preprocessing.image import ImageDataGenerator


class DataLoader():
    def get_datagen(self, key, params):
        datagen = ImageDataGenerator(
            **params[key]["augmentation"]
        )
        generator = datagen.flow_from_directory(
            params[key]["path"],
            **params["generators"][key]
        )
        return generator
