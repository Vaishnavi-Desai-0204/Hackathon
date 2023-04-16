import tensorflow
import numpy as np
import pandas as pd
import os
import re
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.layers import GlobalMaxPooling2D
from keras.applications import ResNet50
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#print(model.summary())

def extract_features(img_path,model):
    img = keras.preprocessing.image.load_img(img_path,target_size=(224,224))
    img_array = keras.preprocessing.image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

filenames = []

for file in os.listdir('fashion-dataset/images'):
    filenames.append(os.path.join('fashion-dataset/images',file))

feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))

pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))

