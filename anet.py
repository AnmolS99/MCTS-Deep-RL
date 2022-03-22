import tensorflow as tf


class ANet:
    """
    Actor network
    """

    def __init__(self, nn_specs, lr) -> None:
        self.lr = lr
        self.nn = self.create_nn(nn_specs)

    def create_nn(self, nn_specs):
        """
        Creating a neural network according to the specifications
        """
        # Converting specs from tuple to list
        nn_specs_list = list(nn_specs)

        # Creating a list of layers
        layers = []

        # Adding input layer
        input_neurons = nn_specs_list[0]
        layers.append(tf.keras.layers.Input((input_neurons, )))

        # Adding all hidden layers
        for layer_neurons in nn_specs_list[1:-1]:
            layers.append(
                tf.keras.layers.Dense(layer_neurons, activation="relu"))

        # Adding output layer, with softmax activation
        output_neurons = nn_specs_list[-1]
        layers.append(
            tf.keras.layers.Dense(output_neurons,
                                  activation=tf.keras.activations.softmax))

        # Creating the neural network model
        model = tf.keras.Sequential(layers)

        # Selecting the optimizer (Adam seems to be the best with adaptive learning rate)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # Compiling the model with categorical cross entropy loss function
        model.compile(optimizer, tf.keras.losses.CategoricalCrossentropy())
        return model
