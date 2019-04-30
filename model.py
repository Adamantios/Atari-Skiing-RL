from keras import Input, Model
from keras.layers import Lambda, Conv2D, Flatten, Dense, Multiply
from keras.backend import cast


def atari_skiing_model(shape, action_size, optimizer) -> Model:
    inputs = Input(shape, name='inputs')
    actions_input = Input((action_size,), name='action_mask')

    normalized = Lambda(lambda x: x / 255.0, name='norm')(inputs)

    conv_1 = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(normalized)
    conv_2 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv_1)
    conv_flattened = Flatten()(conv_2)
    hidden = Dense(256, activation='relu')(conv_flattened)
    output = Dense(action_size)(hidden)
    filtered_output = Multiply(name='QValue')([output, actions_input])

    model = Model(inputs=[inputs, actions_input], outputs=filtered_output)
    model.compile(optimizer, loss=huber_loss)
    return model


def huber_loss(a, b):
    error = a - b
    quadratic_term = error * error / 2
    linear_term = abs(error) - 1 / 2
    use_linear_term = (abs(error) > 1.0)
    # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
    use_linear_term = cast(use_linear_term, 'float32')

    return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term
