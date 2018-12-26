# Introducción a las redes neuronales

La meta de una red neuronal es aproximar comportamientos que encontramos en la naturaleza, de esta manera, podemos traducir problemas del mundo real e intentar resolverlos con ellas.

Lo primero que debemos preguntarnos es: Como representar una neurona física, en el mundo digital, es decir con código. Básicamente debemos preguntarnos como esta percibe sus "Entradas" o inputs y nos da una "Salida" u output correspondiente.

Dicho esto, podemos concluir que una red neuronal artificial es solo un aproximador universal de funciones, lo que significa que podemos entrenarla para resolver muchos problemas complicados del mundo real, aunque la mayor desventaja de las mismas, es que debemos encontrar una manera de optimizar el entrenamiento, de manera que este sea posible en un tiempo razonable.

## El Perceptrón

Es la red neuronal más básica que podemos encontrar, la cual consiste de una sola "neurona", al aprender sobre esta primera aproximación podremos entender como las "neuronas" actúan e interactúan entre sí.

Frank Rosenblatt fue el inventor de este concepto en 1957, así que gracias a él tenemos un modelo que sirve como fundamento básico de este gran mundo.

### Como funciona

Básicamente tenemos una sola neurona con _x<sub>0</sub>_ y _x<sub>1</sub>_ como entradas, de una función que realiza una tarea, en este caso un proceso matemático que nos da _y_ como salida.

El primer ejemplo en el cual podemos pensar cuando hablamos del perceptrón es un clasificador, al cual podemos darle un punto  _(x, y)_ y nos dará un resultado dentro de una clasificación _(A, B)_, hecho esto podremos usar  _aprendizaje supervisado_ para ajustar la salida, básicamente lo que hacemos es darle al perceptrón una entrada de la cual sabemos la respuesta correcta, inicialmente nos dará un resultado, si es que la clasificación es acertada, no hace nada, caso contrario los pesos de las conexiones en las neuronas serán ajustados mediante un proceso matemático para mejorar la clasificación, así que la siguiente iteración se acercara cada vez más a la respuesta correcta, este proceso de _ajuste de pesos (para este caso particular)_ será uno llamado **descenso de gradiente** o **gradient descent**.

Como hemos visto antes, cada entrada al perceptrón tiene un peso, lo que significa que cada conexión tendrá un valor asignado a ella.
En este caso lo que hará el perceptrón es multiplicar los _pesos_ por las entradas y calcular la suma de esos resultados, así:
_<center> x<sub>0</sub> * w<sub>0</sub> + x<sub>1</sub> * w<sub>1</sub> + ... +  x<sub>n</sub> * w<sub>n</sub></center>_

Después necesitamos llamar a una _función de activación_, la cual nos devuelve el resultado en un rango deseado, si comparamos este proceso con la naturaleza, podemos ver las semejanzas que tiene con la sinapsis en el cerebro.

Para nuestro primer ejemplo, podemos hacer una función de activación muy simple, como la función signo: _sign(n)_, la cual toma un número como la entrada y evalúa la siguiente expresión: _if n > 0 then return 1, else return -1_, hasta ahora todo este proceso en conjunto _(suma de pesos y función de activación)_ es llamado **feed forward**

En este ejemplo [simplePerceptron.py](/sources/nn-lib/Perceptron/simplePerceptron.py), vamos a tomar la función de una recta: _f(x) = x_, cada punto que este sobre la recta será clasificado con 1, mientras que cada punto que este debajo será clasificado con -1, de esta manera, como ya sabemos que problema vamos a resolver, podemos crear nuestro propio _Set de datos de entrenamiento_ que no es nada más que pares ordenados _(x, y)_ que ya sabemos a qué clasificación pertenecen.
