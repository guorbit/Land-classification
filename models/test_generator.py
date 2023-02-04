from constructor import ModelGenerator

if __name__ == "__main__":
    model = ModelGenerator("test", (256,256,3), (256,256,3), 2)
    print(model.name)
