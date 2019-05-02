from keras import Input, Model
from keras.layers import Lambda, Conv2D, Flatten, Dense, Multiply
from keras.backend import cast
from keras.optimizers import Optimizer


def min_frame_dim_that_passes_net() -> int:
    """
    Calculates the minimum that each frame's dimension is required to have,
    in order to pass through the network.

    :return: the min frame dimension.
    """
    # (last conv size + filter loss) * first conv stride, or first conv size if it is bigger.
    # ( 4 + 1 ) * 4 or 8
    return 20


def frame_can_pass_the_net(height, width) -> bool:
    """
    Returns if a frame can successfully pass through the network.

    :param height: the frame's height.
    :param width: the frame's width.
    :return: bool.
    """
    return height >= min_frame_dim_that_passes_net() and width >= min_frame_dim_that_passes_net()


def atari_skiing_model(shape: tuple, action_size: int, optimizer: Optimizer) -> Model:
    """
    Defines a Keras Model designed for the atari skiing game.

    :param shape: the input shape.
    :param action_size: the number of available actions.
    :param optimizer: an optimizer to be used for model compilation.
    :return: the Keras Model.
    """
    # Create the input layers.
    inputs = Input(shape)
    actions_input = Input((action_size,))
    # Create a normalization layer.
    normalized = Lambda(lambda x: x / 255.0)(inputs)
    # Create hidden layers.
    conv_1 = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(normalized)
    conv_2 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv_1)
    conv_flattened = Flatten()(conv_2)
    dense = Dense(256, activation='relu')(conv_flattened)
    # Create and filter the output, multiplying it with the actions input mask, in order to get the QTable.
    output = Dense(action_size)(dense)
    filtered_output = Multiply()([output, actions_input])

    # Create the model.
    model = Model(inputs=[inputs, actions_input], outputs=filtered_output)
    # Compile the model.
    model.compile(optimizer, loss=huber_loss)

    return model


def huber_loss(y_true, y_pred):
    """
    Define the huber loss.

    :param y_true: the true value.
    :param y_pred: the predicted value.
    :return: a tensor with the result.
    """
    # Calculate the error.
    error = y_true - y_pred

    # Calculate MSE.
    quadratic_term = error * error / 2
    # Calculate MAE.
    linear_term = abs(error) - 1 / 2

    # Use mae if |error| > 1.
    use_linear_term = (abs(error) > 1.0)
    # Cast the booleans to floats, in order to be compatible with Keras.
    use_linear_term = cast(use_linear_term, 'float32')

    # Return MAE or MSE based on the flag.
    return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term
