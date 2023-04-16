import tarfile
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf


# Extract the Polyvore dataset from the .tar.gz file
tar = tarfile.open('polyvore.tar.gz', 'r:gz')
tar.extractall()
tar.close()

# Load the trained CNN model
model = tf.keras.models.load_model('polyvore_cnn.h5')

# Load the user's wardrobe
wardrobe = np.load('wardrobe.npy')

# Encode the clothes in the user's wardrobe as feature vectors
wardrobe_features = model.predict(wardrobe)

# Choose a piece of clothing from the user's wardrobe
chosen_clothing = wardrobe[0]
chosen_feature = wardrobe_features[0]

# Compute the cosine similarity between the chosen clothing and all the clothes in the Polyvore dataset
polyvore_images = np.load('polyvore/images.npy')
polyvore_features = model.predict(polyvore_images)
similarities = cosine_similarity(chosen_feature.reshape(1, -1), polyvore_features)

# Sort the outfits in the Polyvore dataset based on their similarity to the chosen clothing
outfit_indices = np.argsort(similarities[0])[::-1]
top_outfits = polyvore_images[outfit_indices[:5]]

# Display the top outfits to the user
# You can use a library like OpenCV or Matplotlib to display the images
