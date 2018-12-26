
# Introduction to neural networks

The goal of neural networks is to resemble natural behaviors, so we can translate real world problems and try to solve with them.

The first thing we need to ask is how we represent a neuron in code, and how we use it, basically we need to understand how it receives inputs and gives a corresponding output.

We can say that a neural network is just a universal function approximation, which means that we could train it to solve most of known problems; the problem is that we need to come up with some clever ways to optimize our training, and make it in a way that it will be doable in a reasonable period of time.

## The Perceptron

Is the most basic of neural networks, consist of only one "neuron", it will help us to understand how "neurons" act and interact.

Frank Rosenblatt invented this concept in 1957, so thanks to him we have a model that is the basis of this great world.

### How it works

Basically we have a single neuron with _x<sub>0</sub>_ and _x<sub>1</sub>_ as inputs that go into a function which performs a task or a mathematical process and gives us _y_ as the output

The first example we can think of when we talk about the perceptron is a classifier, so we can give it a point _(x, y)_ and it will return us a classification _(A, B)_ then we will use _supervised learning_ to adjust the output, basically in supervised learning we give the perceptron an input to which we know the correct answer, then it will classify it, if it guesses correctly it does nothing, but if the answer is wrong we make it tweak some of the math that goes behind _(adjust weights)_ it so the next time it will get closer to the correct answer, this _tweak (for this particular case)_ is called **gradient descent**.

As we have seen before, each input on the perceptron goes weighted, which means that each connection has a value assigned to it.
What the perceptron does is multiply this _weights_ with the inputs, and then compute the sum of all of those results:
_<center> x<sub>0</sub> * w<sub>0</sub> + x<sub>1</sub> * w<sub>1</sub> + ... +  x<sub>n</sub> * w<sub>n</sub></center>_

Then we need to call the _activation function_ which conforms the output to a desire range, if we compare this process, we see that it resembles the synapsis process on the brain.

For a first example we can make a very simple activation function, like a sign function _sign(n)_ which takes a number as the input, and for the output it evaluates: _if n > 0 then return 1, otherwise return -1_. So all of this process _(weight sum, and activation function)_ it is called **feed forward**.

On this example [simplePerceptron.py](/sources/nn-lib/Perceptron/simplePerceptron.py), we are going to take a line function _f(x) = x_, every point above the line will be classified as a 1, everything below it should be a -1, knowing the problem that we are going to solve, we can create our own _training Dataset_ which is nothing more than some points _(x, y)_ we already know where they belong
