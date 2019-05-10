from typing import Union

from keras import Input, Model
from keras.backend import cast
from keras.layers import Lambda, Flatten, Dense, Multiply, ConvLSTM2D, BatchNormalization
from keras.optimizers import Optimizer, adam, rmsprop, sgd, adagrad, adadelta, adamax

# (last conv size + filter loss) * first conv stride, or first conv size if it is bigger.
# ( 4 + 1 ) * 4 or 8
MIN_FRAME_DIM_THAT_PASSES_NET = 20


def frame_can_pass_the_net(height: int, width: int) -> bool:
    """
    Returns if a frame can successfully pass through the network.

    :param height: the frame's height.
    :param width: the frame's width.
    :return: bool.
    """
    return height >= MIN_FRAME_DIM_THAT_PASSES_NET and width >= MIN_FRAME_DIM_THAT_PASSES_NET


def atari_skiing_model(shape: tuple, action_size: int, optimizer: Optimizer) -> Model:
    """
    Defines a Keras Model designed for the atari skiing game.

    :param shape: the input shape.
    :param action_size: the number of available actions.
    :param optimizer: an optimizer to be used for model compilation.
    :return: the Keras Model.
    """
    # Create the input layers.
    inputs = Input(shape, name='input')
    actions_input = Input((action_size,), name='input_mask')
    # Create a normalization layer.
    normalized = Lambda(lambda x: x / 255.0, name='normalisation')(inputs)

    # Create CNN-LSTM layers.
    conv_lstm2d_1 = ConvLSTM2D(16, (8, 8), strides=(4, 4), activation='relu', return_sequences=True,
                               name='conv_lstm_2D_1')(normalized)
    batch_norm = BatchNormalization(name='batch_norm1')(conv_lstm2d_1)
    conv_lstm2d_2 = ConvLSTM2D(32, (4, 4), strides=(2, 2), activation='relu', name='conv_lstm_2D_2')(batch_norm)
    batch_norm2 = BatchNormalization(name='batch_norm2')(conv_lstm2d_2)

    # Flatten the output and pass it to a dense layer.
    flattened = Flatten(name='flatten')(batch_norm2)
    dense = Dense(256, activation='relu', name='dense1')(flattened)

    # Create and filter the output, multiplying it with the actions input mask, in order to get the QTable.
    output = Dense(action_size, name='dense2')(dense)
    filtered_output = Multiply(name='filtered_output')([output, actions_input])

    # Create the model.
    model = Model(inputs=[inputs, actions_input], outputs=filtered_output)
    # Compile the model.
    model.compile(optimizer, loss=huber_loss)

    return model


def initialize_optimizer(optimizer_name: str, learning_rate: float, beta1: float, beta2: float,
                         lr_decay: float, rho: float, fuzz: float, momentum: float) \
        -> Union[adam, rmsprop, sgd, adagrad, adadelta, adamax]:
    """
    Initializes an optimizer based on the user's choices.

    :param optimizer_name: the optimizer's name.
        Can be one of 'adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adamax'.
    :param learning_rate: the optimizer's learning_rate
    :param beta1: the optimizer's beta1
    :param beta2: the optimizer's beta2
    :param lr_decay: the optimizer's lr_decay
    :param rho: the optimizer's rho
    :param fuzz: the optimizer's fuzz
    :param momentum: the optimizer's momentum
    :return: the optimizer.
    """
    if optimizer_name == 'adam':
        return adam(lr=learning_rate, beta_1=beta1, beta_2=beta2, decay=lr_decay)
    elif optimizer_name == 'rmsprop':
        return rmsprop(lr=learning_rate, rho=rho, epsilon=fuzz)
    elif optimizer_name == 'sgd':
        return sgd(lr=learning_rate, momentum=momentum, decay=lr_decay)
    elif optimizer_name == 'adagrad':
        return adagrad(lr=learning_rate, decay=lr_decay)
    elif optimizer_name == 'adadelta':
        return adadelta(lr=learning_rate, rho=rho, decay=lr_decay)
    elif optimizer_name == 'adamax':
        return adamax(lr=learning_rate, beta_1=beta1, beta_2=beta2, decay=lr_decay)
    else:
        raise ValueError('An unexpected optimizer name has been encountered.')


def huber_loss(y_true, y_pred):
    """
    Define the huber loss.

    :param y_true: the true value.
    :param y_pred: the predicted value.
    :return: a tensor with the result.
    """
    # Calculate the error.
    error = abs(y_true - y_pred)

    # Calculate MSE.
    quadratic_term = error * error / 2
    # Calculate MAE.
    linear_term = error - 1 / 2

    # Use mae if |error| > 1.
    use_linear_term = (error > 1.0)
    # Cast the boolean to float, in order to be compatible with Keras.
    use_linear_term = cast(use_linear_term, 'float32')

    # Return MAE or MSE based on the flag.
    return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term
