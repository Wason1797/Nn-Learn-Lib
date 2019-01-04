from nn_lib.samples.perceptron_samples import simple_perceptron_sample
from nn_lib.samples.perceptron_samples import refined_perceptron_sample
from nn_lib.Layers import nn


def render_menu():
    print("NN-Learn-lib examples: ")
    print("---------------------- ")
    print("1. Simple perceptron example with f(x) = x")
    print("2. Refined perceptron example with f(x) = mx+b")
    print("q. Exit")


# while True:
#     render_menu()
#     opt = input("Select an option from the avove ... ")
#     if opt is "1":
#         simple_perceptron_sample.run()
#     elif opt is "2":
#         refined_perceptron_sample.run()
#     elif opt is "q":
#         break

network = nn.NeuralNetwork(3, 5, 2)

network.train([1, 2, 3], [1, 2], 0.1)
