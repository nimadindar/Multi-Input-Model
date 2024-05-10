import tensorflow as tf 
import tensorflow_addons as tfa

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



def create_cfg_from_file(filename):
    import imp
    f = open(filename)
    # global data
    data = imp.load_source('data', filename, f)
    for key in [data.concat_activation, data.pooling_activation, data.input_activation, data.hidden_activation, 
        data.concat_activation, data.concat_hidden_activation]:
        for i in range(len(key)):
            if key == "relu":
                key[i] = activation_dict["ReLU"]
            elif key[i] == "gelu":
                key[i] = activation_dict["GELU"]
            elif key[i] == "elu":
                key[i] = activation_dict["ELU"]  
            elif key[i] == "selu":
                key[i] = activation_dict["SELU"]
            elif key[i] == "softplus":
                key[i] = activation_dict["SoftPlus"]
            elif key[i] == "swish":
                key[i] = activation_dict["Swish"]
            elif key[i] == "mish":
                key[i] = activation_dict["Mish"]
            elif key[i] == "tanh":
                key[i] = activation_dict["Tanh"]
            elif key[i] == "sigmoid":
                key[i] = activation_dict["Sigmoid"]
            else:
                key[i] = activation_dict["Linear"] 
    f.close()
    return data
