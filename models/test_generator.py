from constructor import ModelGenerator, VGG16_UNET

if __name__ == "__main__":
    model = VGG16_UNET((512,512,3), 2)
    print(model.name)
    model.create_model()
    print(model.summary())
