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
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

plt.rcParams['font.size'] = 16
path = 'fashion-dataset/images'
images_df = pd.read_csv('fashion-dataset/images.csv')
styles_df = pd.read_csv('fashion-dataset/styles.csv', on_bad_lines='skip')

styles_df['filename'] = styles_df['id'].astype(str) + '.jpg'

image_files = os.listdir(path)

styles_df['present'] = styles_df['filename'].apply(lambda x: x in image_files)

styles_df = styles_df[styles_df['present']].reset_index(drop=True)

styles_df = styles_df.sample(10000)

img_size = 224
datagen = ImageDataGenerator(rescale=1/255.)
generator = datagen.flow_from_dataframe(dataframe = styles_df,
                                      directory = path,
                                      target_size = (img_size, img_size),
                                      x_col = 'filename',
                                      class_mode = None,
                                      batch_size = 32,
                                      shuffle = False,
                                      classes = None)

base_model = VGG16(include_top = False, input_shape = (img_size, img_size,3))

for layer in base_model.layers:
    layer.trainable = False 
    
input_layer = Input(shape = (img_size, img_size, 3))
x = base_model(input_layer)
output = GlobalAveragePooling2D()(x)

embeddings = Model(inputs = input_layer, outputs = output)
embeddings.summary

X = embeddings.predict(generator, verbose = 1)

pca = PCA(2)
X_pca = pca.fit_transform(X)

styles_df[['pc1','pc2']] = X_pca

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, styles_df['masterCategory'])

# ask user to input the path of the image
img_path = input('Enter the path of the image: ')

# load the image
img = Image.open(img_path)
img = img.resize((img_size, img_size))
img_array = np.array(img) / 255.

# generate the embeddings for the image
img_emb = embeddings.predict(np.array([img_array]), verbose=0)

# reduce the dimensionality of the embeddings
img_pca = pca.transform(img_emb)

# predict the category using KNN
category = knn.predict(img_emb)[0]

# display the image and its predicted category
plt.figure(figsize=(6,6))
plt.imshow(img_array)
plt.title(f'Predicted Category: {category}')
plt.axis('off')
plt.show()





