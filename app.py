import pickle                                                                             ##used for serializing and deserializing Python objects.
import tensorflow                                                                          ## used for building and training machine learning models.
from tensorflow.keras.preprocessing import image                                            ## provides functions for loading and processing images.
from tensorflow.keras.layers import GlobalMaxPooling2D                                    ## reduce the spatial dimensions of the input while retaining the most important features.
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input               ##ResNet50 is a pre-trained deep learning model, and preprocess_input prepares images for the model by scaling pixel values.
import numpy as np                                                                        ##  provides support for numerical operations and array manipulations.
from numpy.linalg import norm                                                              ## used to calculate the norm of vectors (used for normalization).
import os                                                                                    ##provides functions for interacting with the operating system
from tqdm import tqdm                                                                        ##  typically used to visualize the progress of loops.

model = ResNet50(weights='imagenet',include_top=False, input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([                                ## This layer reduces the dimensionality of the output from the convolutional layers while preserving the most significant features.
    model,
    GlobalMaxPooling2D(),
])

def extract_features(img_path , model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array,axis=0)                ## Expands the dimensions of the image array to create a batch of one image
    preprocessed_img = preprocess_input(expanded_img_array)            ##  which scales pixel values to the range expected by the model 
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)                            ## The result is then flattened into a one-dimensional array.

    return normalized_result

filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))                ## Appends the full file path of each image to the filenames list by joining the directory name with the file name.

feature_list =[]

for file in filenames:
    feature_list.append(extract_features(file,model))               ## Calls the extract_features function for each image file and appends the resulting feature vector to the feature_list.


# print(np.array(feature_list).shape) #                            ## print the shape of the feature_list array, providing information about the number of images and the size of the feature vectors.
pickle.dump(feature_list,open('feature_list.p','wb'))                ## Serializes the feature_list and saves it to a file named feature_list.p in binary write mode using the pickle module.
pickle.dump(filenames,open('filenames.p','wb'))
