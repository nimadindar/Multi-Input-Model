import tensorflow as tf 
import tensorflow_addons as tfa
import numpy as np
from keras.constraints import max_norm
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense,Dropout, AveragePooling2D
from keras import mixed_precision
from sklearn.metrics import classification_report, confusion_matrix

from cfg import create_cfg_from_file

import os
import random


import argparse

# Define the argument parser
# Argparse gets its parameters from run_models.py
# model.py save/loads models based on the --mode (train/test) and the testId
parser = argparse.ArgumentParser(description='My Python Script')
parser.add_argument('--mode', type=str, required=True, help='The mode should be either train or test.')
parser.add_argument('--test_number', type=str, required=True, help='The ID of test.')

# Get the argument values
args = parser.parse_args()


SEED = 10

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

set_seeds(seed=SEED)

# Model env Config
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

mixed_precision.set_global_policy('mixed_float16')

cwd = os.path.dirname(os.path.abspath(__file__))+'/'
cwd = cwd.replace('\\','/')

# Global Parameters
excluded_pool_layer_ratio=0.4
excluded_pool_layer_default=2


results_folder = 'Results'


activation_dict = {"ReLU": tf.keras.activations.relu, 
                   "GELU":tf.keras.activations.gelu, 
                   "ELU":tf.keras.activations.elu, 
                   "SELU":tf.keras.activations.selu, 
                   "SoftPlus":tf.keras.activations.softplus, 
                   "Swish":tf.keras.activations.swish, 
                   "Mish":tfa.activations.mish,
                   "Tanh": tf.keras.activations.tanh,
                   "Linear": tf.keras.activations.linear,
                   "Sigmoid": tf.keras.activations.sigmoid}

# cfg loads an instance of your model parameters and passes to the model 
cfg = create_cfg_from_file(cwd+results_folder+f"/cfgs/test{args.test_number}.txt")


# define your data load
class load_data():

    def __init__(self):
        pass
    # Load your data

# This is a dummy load of data - Replace it with your own data load
xtrain, xtrain2, ytrain = load_data.load_data()
xtest, xtest2, ytest = load_data.load_data()


def model(cfg, input_shape_conv,input_shape_mlp):

    """
    model() function recieves your model configuration and shape of your data to create a multi-input model.

    model has one input for CNN and another input for MLP, then the ouput of these model are concatenated 
    and passed to another MLP to make final predictions.

    CNN:
        CNN layer (2D) ----> Max/Average Pooling layer (2D) ----> Dropout ----> Pooling Activation layer
    
    MLP:
        Dense ----> Dropout ----> Hidden Activation Layer
    """

    "CNN model (Input 1)"

    inputs_x = Input(shape =input_shape_conv)
    x = tf.keras.layers.BatchNormalization()(inputs_x)

    for num_layer in range(len(cfg.list_filters)):

        x = Conv2D(filters = cfg.list_filters[num_layer],
                kernel_size= cfg.list_kernels[num_layer], 
                activation=cfg.conv_activation[num_layer], padding='same')(x)
        
        if 12 < len(cfg.list_filters):
            excluded_pool_layer = np.round(excluded_pool_layer_ratio*len(cfg.list_filters))
        elif 11 < len(cfg.list_filters) <= 12:
            excluded_pool_layer = excluded_pool_layer_default+2        
        elif 10 < len(cfg.list_filters) <= 11:
            excluded_pool_layer = excluded_pool_layer_default+1
        elif 5 < len(cfg.list_filters) <= 10:
            excluded_pool_layer = excluded_pool_layer_default
        else:
            excluded_pool_layer = excluded_pool_layer_default-1

        if num_layer < len(cfg.list_filters) - excluded_pool_layer:
            if cfg.pooling[num_layer] == "max":
                x = MaxPooling2D(pool_size=(2,2))(x)
            else:
                x = AveragePooling2D(pool_size=(2,2))(x)

        x = Dropout(cfg.conv_dropout[num_layer])(x)
        x = cfg.conv_activation[num_layer](x)

    if cfg.global_pool == 'avg':
        x_output = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif cfg.global_pool == 'max':
        x_output = tf.keras.layers.GlobalMaxPooling2D()(x)
    else:
        x_output = Flatten()(x)

    cnn_model = Model(inputs_x, x_output)

    "MLP model (Input 2)"
    inputs_y = Input(shape=input_shape_mlp)
    y = tf.keras.layers.BatchNormalization()(inputs_y)

    for num_layer in range(len(cfg.input_neurons)):
        y = Dense(units = cfg.input_neurons[num_layer], 
                activation=cfg.input_activation[num_layer], 
                kernel_constraint=max_norm(cfg.kerenel_constraint))(y)

        y = Dropout(cfg.input_dropout[num_layer])(y)

        if(num_layer < (len(cfg.input_neurons)-1)):
            y = cfg.hidden_activation[num_layer](y)

    y_output = cfg.hidden_activation[num_layer](y)

    mlp_model = Model(inputs_y, y_output)


    "MLP model (concatenation)"

    z = tf.keras.layers.concatenate([cnn_model.output,mlp_model.output])

    for num_layer in range(len(cfg.concat_neurons)):
        z = Dense(units = cfg.concat_neurons[num_layer], 
                activation=cfg.concat_activation[num_layer], 
                kernel_constraint=max_norm(cfg.kerenel_constraint))(z)

        if(num_layer < (len(cfg.concat_neurons)-1)):
            z = Dropout(cfg.concat_dropout[num_layer])(z)    
            z = cfg.concat_hidden_activation[num_layer](z)
        
    z_output = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(z)

    model = tf.keras.Model(inputs=[cnn_model.input, mlp_model.input], outputs = z_output)
    model.save(cwd+f'{results_folder}/model{args.test_number}.h5')

    return model

# Instantiate CFG here!
input_shape_conv = ()
input_shape_mlp = ()

model = model(cfg, input_shape_conv, input_shape_mlp)
optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
optimizer = mixed_precision.LossScaleOptimizer(optimizer)
model.compile(optimizer= optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

checkponit = tf.keras.callbacks.ModelCheckpoint(filepath=cwd+f'{results_folder}/model{args.test_number}_checkpoint.h5', 
                                                monitor='val_accuracy', save_best_only=True)

if args.mode == 'train':

    hist_callback = tf.keras.callbacks.TensorBoard(
    log_dir= cwd+f'{results_folder}/model{args.test_number}/',
    histogram_freq=1,
    # embeddings_freq=1,
    embeddings_freq=0,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq='epoch')

    history = model.fit(
        [xtrain,xtrain2], 
        ytrain,
        epochs=cfg.num_epochs,
        callbacks=[checkponit, hist_callback],
        validation_data=([xtest,xtest2], ytest),
        batch_size=cfg.batch_size,
        verbose=2
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(acc) + 1)

    with open(cwd+f'{results_folder}/accuracy_history_model{args.test_number}.txt', 'w') as file:
        for epoch, (train_accuracy, val_accuracy) in enumerate(zip(acc, val_acc)):
            file.write(f"Epoch {epoch + 1}: Train Accuracy - {train_accuracy}, Validation Accuracy - {val_accuracy}\n")

else:
    load_model = tf.keras.models.load_model(cwd+f'{results_folder}/model{args.test_number}.h5')
    model = tf.keras.models.clone_model(load_model)
    model.load_weights(filepath=cwd+f'{results_folder}/model{args.test_number}_checkpoint.h5')

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(optimizer= optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    ytest_pred = model.predict([xtest,xtest2])
    ytest_pred = np.argmax(ytest_pred, axis=1)
    print(ytest_pred.shape)
    cm = confusion_matrix(ytest, ytest_pred)
    print(cm)
    class_report = classification_report(ytest,ytest_pred,digits=4)
    with open(cwd+f'{results_folder}/classRep_model{args.test_number}.txt', 'w') as outfile:
        outfile.write(class_report)
    print(class_report)
    print('\n')
    np.savetxt(cwd+f'{results_folder}/confusion_matrix_model{args.test_number}.txt', cm, fmt='%d')