# By Abtin Riasatian, email: abtin.riasatian@uwaterloo.ca

# The extract_features function gets a patch directory and a feature directory.
# the function will extract the features of the patches inside the folder
# and saves them in a pickle file of dictionary mapping patch names to features.


# config variables ---------------------------------------------
patch_dir = "./patches/"
extracted_features_save_adr = "./extracted_features.pickle"
network_weights_address = "./weights/KimiaNetKerasWeights.h5"
network_input_patch_width = 1000
batch_size = 30
img_format = 'jpg'
use_gpu = True
# =============================================================


# importing libraries----------------------------------------------------
import os

if use_gpu:
    os.environ['NVIDIA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Lambda
# from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.backend import bias_add, constant    

import glob, pickle, skimage.io, pathlib
import numpy as np
from tqdm import tqdm
# ========================================================================


# feature extractor preprocessing function
def preprocessing_fn(input_batch, network_input_patch_width):

    org_input_size = tf.shape(input_batch)[1]
    
    # standardization
    scaled_input_batch = tf.cast(input_batch, 'float') / 255.
    
    
    # resizing the patches if necessary
    resized_input_batch = tf.cond(tf.equal(org_input_size, network_input_patch_width),
                                lambda: scaled_input_batch, 
                                lambda: tf.image.resize(scaled_input_batch, 
                                                        (network_input_patch_width, network_input_patch_width)))
    
    
    # normalization, this is equal to tf.keras.applications.densenet.preprocess_input()---------------
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_format = "channels_last"
    mean_tensor = constant(-np.array(mean))
    standardized_input_batch = bias_add(resized_input_batch, mean_tensor, data_format)
    standardized_input_batch /= std
    #=================================================================================================
    
    return standardized_input_batch

# feature extractor initialization function
def kimianet_feature_extractor(network_input_patch_width, weights_address):
    
    dnx = DenseNet121(include_top=False, weights=weights_address, 
                      input_shape=(network_input_patch_width, network_input_patch_width, 3), pooling='avg')

    kn_feature_extractor = Model(inputs=dnx.input, outputs=GlobalAveragePooling2D()(dnx.layers[-3].output))
    
    kn_feature_extractor_seq = Sequential([Lambda(preprocessing_fn, 
                                                  arguments={'network_input_patch_width': network_input_patch_width}, 
                                   input_shape=(None, None, 3), dtype=tf.uint8)])
    
    kn_feature_extractor_seq.add(kn_feature_extractor)
    
    return kn_feature_extractor_seq

# feature extraction function
def extract_features(patch_dir, extracted_features_save_adr, network_weights_address, 
                     network_input_patch_width, batch_size, img_format):
        
    feature_extractor = kimianet_feature_extractor(network_input_patch_width, network_weights_address)
    
    patch_adr_list = [pathlib.Path(x) for x in glob.glob(patch_dir+'*.'+img_format)]
    feature_dict = {}

    for batch_st_ind in tqdm(range(0, len(patch_adr_list), batch_size)):
        batch_end_ind = min(batch_st_ind+batch_size, len(patch_adr_list))
        batch_patch_adr_list = patch_adr_list[batch_st_ind:batch_end_ind]
        patch_batch = np.array([skimage.io.imread(x) for x in batch_patch_adr_list])
        batch_features = feature_extractor.predict(patch_batch)
        feature_dict.update(dict(zip([x.stem for x in batch_patch_adr_list], list(batch_features))))
        
        with open(extracted_features_save_adr, 'wb') as output_file:
            pickle.dump(feature_dict, output_file, pickle.HIGHEST_PROTOCOL)


extract_features(patch_dir, extracted_features_save_adr, network_weights_address, 
                 network_input_patch_width, batch_size, img_format)

