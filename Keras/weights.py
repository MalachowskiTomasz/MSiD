from keras.models import load_model

def main():
    model = load_model("Bmodel2.h5")
    print(model.get_weights()[0][0])

if __name__ == '__main__':
    main()