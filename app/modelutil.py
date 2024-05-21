import os 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
import tensorflow as tf

def load_model() -> Sequential: 
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

   # model.load_weights(os.path.join('..','models','checkpoint'))
    #checkpoint_dir = os.path.join('..', 'models')
    checkpoint_dir = os.path.join('workspaces', 'LipNet', 'models')

    # Create a checkpoint instance that points to the folder where the checkpoints are saved
    checkpoint = tf.train.Checkpoint(model=model)

    # Restore the latest checkpoint
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        checkpoint.restore(latest)
    else:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    return model
