from nn_lib.samples.perceptron_samples import simple_perceptron_sample
from nn_lib.samples.perceptron_samples import refined_perceptron_sample
from nn_lib.NeuralNetworkRep.NeuralNetworksViz import viz as nn_viz
from nn_lib.NeuralNetworkRep.NeuralNetworksAnimation import viz as nn_animation


def render_menu():
    print("NN-Learn-lib examples: ")
    print("---------------------- ")
    print("1. Simple perceptron example with f(x) = x")
    print("2. Refined perceptron example with f(x) = mx+b")
    print("3. Neural Network graphic representation")
    print("4. Neural Network animation")
    print("q. Exit")


while True:
    render_menu()
    opt = input("Select an option from the avove ... ")
    if opt is "1":
        simple_perceptron_sample.run()
    elif opt is "2":
        refined_perceptron_sample.run()
    elif opt is "3":
        nn_viz.main()
    elif opt is "4":
        nn_animation.main()
    elif opt is "q":
        break
