import tensorflow as tf


class ANet:
    """
    Actor network
    """

    def __init__(self, nn_specs, lr, optimizer) -> None:
        self.lr = lr
        self.optimizer = optimizer
        self.nn = self.create_nn(nn_specs)

    def create_nn(self, nn_specs):
        """
        Creating a neural network according to the specifications
        """

        # Creating a list of layers
        layers = []

        # Adding input layer
        input_neurons = nn_specs[0]
        layers.append(tf.keras.layers.Input((input_neurons, )))

        # Adding all hidden layers
        for layer_neurons in nn_specs[1:-1]:
            layers.append(
                tf.keras.layers.Dense(layer_neurons[0],
                                      activation=self.parse_act_func(
                                          layer_neurons[1])))

        # Adding output layer, with softmax activation
        output_neurons = nn_specs[-1]
        layers.append(
            tf.keras.layers.Dense(output_neurons,
                                  activation=tf.keras.activations.softmax))

        # Creating the neural network model
        model = tf.keras.Sequential(layers)

        # Parsing the optimizer (Adam seems to be the best with adaptive learning rate)
        optimizer = self.parse_optimizer(self.optimizer)

        # Compiling the model with categorical cross entropy loss function
        model.compile(optimizer, tf.keras.losses.CategoricalCrossentropy())

        return model

    def parse_optimizer(self, optimizer):
        """
        Parsing optimizer
        """
        if optimizer.lower() == "adagrad":
            return tf.keras.optimizers.Adagrad(learning_rate=self.lr)
        elif optimizer.lower() == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=self.lr)
        elif optimizer.lower() == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate=self.lr)
        elif optimizer.lower() == "adam":
            return tf.keras.optimizers.Adam(learning_rate=self.lr)

    def parse_act_func(self, act_func):
        """
        Parsing activation function
        """
        if act_func.lower() == "linear":
            return tf.keras.activations.linear
        elif act_func.lower() == "sigmoid":
            return tf.keras.activations.sigmoid
        elif act_func.lower() == "tanh":
            return tf.keras.activations.tanh
        elif act_func.lower() == "relu":
            return tf.keras.activations.relu