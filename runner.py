from neuralnet import *
import mnist_loader
# from PIL import Image

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = Network([784, 30, 15, 10], activation_function='sigmoid')

net.SGD(training_data, 15, 10, 1, test_data=test_data)

folder = "pictures/test/"

# img = np.array([[x] for x in np.ndarray.flatten(np.array(Image.open(target + "9/7.jpg")))])/255

# result = net.identify(img)

result = net.identify_from_path(folder + "9/7.jpg")

print(result)
