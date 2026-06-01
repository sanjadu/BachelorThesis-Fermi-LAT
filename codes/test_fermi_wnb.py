import os
import math
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

os.environ["KERAS_HOME"] = "/d12/CAC/sd0017/downloads/keras/"

import sklearn
import numpy
import matplotlib as mpl

import wandb
from wandb.integration.keras import WandbMetricsLogger


print("TF version:", tf.__version__)
print("sklearn version: ", sklearn.__version__)
print("matplotlib version: ", mpl.__version__)

# wandb
wandb.init(project="FERMI-LAT", config={"latent_dim":64, "conv4": 64})
config = wandb.config

latent_dim = config.latent_dim #3e-1 # 128 
conv4 = config.conv4 # 64 



base_path = "/d12/CAC/sd0017/downloads/FermiLAT/"


#### load data
import datareader_bkg_src_Asi

###############################
# load the source, bkg and tot lists
###############################

path_to_dir='/d12/CAC/sd0017/downloads/test_im_iem_psr_bll_fsrq_pwn1_2_patch768/'

(arr_list, num_list, arr_files) = datareader_bkg_src_Asi.read_npy_data(path_to_dir, channel=0, sqrt=0, train=True,  norm=False)

print ('check the number of ims in different lists: ', len(num_list))
print ('check shape: ', arr_list[0].shape) # shape here is ()
a = arr_list[0]
arr0 = a
print("full arr shape: ", arr0.shape)
bkg0 = arr0[0,:,:,0]
print("background shape: ", bkg0.shape)

print ('check example total array counts: ', arr_list[0].sum())

print ('check consistency in filenames: ', arr_files[3], arr_files[300], arr_files[497])

arr_list_to_arr = np.array(arr_list)
print(arr_list_to_arr.shape)

bkg_list = []
for i in range(len(arr_list)):
    bkg = arr_list[i][0,:,:,:]
    bkg_list.append(bkg)

bkg_list_to_arr = np.array(bkg_list)
print(bkg_list_to_arr.shape)

files_list_to_arr = np.array(arr_files)


bkg_train, bkg_test, files_train, files_test = sklearn.model_selection.train_test_split(bkg_list_to_arr, files_list_to_arr, test_size=0.2, random_state=20)

print(bkg_train.shape)

bkg_train2 = bkg_train
bkg_test2 = bkg_test



### we are interested in reconstruction
### not to learn labels


# normalize to make sure the array values are always within [0,1]
# first find biggest pixel value for that bkg image and then divide the whole image with that number
bkg_train_norm = []
bkg_train2_norm = []
for i in range(len(bkg_train)):
    max_value_train = np.max(bkg_train[i])
    bkg_train_norm.append(bkg_train[i]/max_value_train)
    bkg_train2_norm.append(bkg_train2[i]/max_value_train)
bkg_train_norm_arr = np.array(bkg_train_norm)
bkg_train2_norm_arr = np.array(bkg_train2_norm)
print(bkg_train_norm_arr.shape)

bkg_test_norm = []
bkg_test2_norm = []
for i in range(len(bkg_test)):
    max_value_test = np.max(bkg_test[i])
    bkg_test_norm.append(bkg_test[i]/max_value_test)
    bkg_test2_norm.append(bkg_test2[i]/max_value_test)
bkg_test_norm_arr = np.array(bkg_test_norm)
bkg_test2_norm_arr = np.array(bkg_test2_norm)



# shape of the input
input_shape = (64, 64, 1)


#### autoencoder as function #######
encoder_inps = layers.Input(shape=input_shape)
def autoencoder(encoder_inps, latent_dim:int, conv4:int):
  x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", strides=2)(encoder_inps)
  # strides make dim/2 16
  x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", strides=1)(x)
  x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", strides=1)(x)
  x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", strides=2)(x) # 8
  x = layers.Conv2D(conv4, (3, 3), activation="relu", padding="same", strides=1)(x)
  x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", strides=1)(x)
  x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", strides=2)(x) # 4
  x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", strides=1)(x)
  x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", strides=1)(x)

  x = layers.Flatten()(x) # 1d array
  bottleneck = layers.Dense(latent_dim, activation="relu", name="bottleneck")(x)

  encoder = models.Model(encoder_inps, bottleneck, name="encoder")
  print ('encoder summary: ', '\n', encoder.summary())


  # decoder
  decoder_inps = layers.Input(shape=(latent_dim,))

  x = layers.Dense(4 * 4 * latent_dim, activation="relu")(decoder_inps)
  x = layers.Reshape((4, 4, latent_dim))(x)

  x = layers.Conv2DTranspose(128, (3, 3), activation="relu", padding="same", strides=1)(x)
  #strides 1 does not change shape
  x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same", strides=1)(x)
  x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same", strides=2)(x)
  x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same", strides=1)(x)
  x = layers.Conv2DTranspose(conv4, (3, 3), activation="relu", padding="same", strides=1)(x)
  x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same", strides=2)(x)
  x = layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same", strides=1)(x)
  x = layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same", strides=1)(x)
  x = layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same", strides=2)(x)

  decoder_outputs = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

  decoder = models.Model(decoder_inps, decoder_outputs, name="decoder")
  print ('only decoder summary: ', '\n', decoder.summary())


  # full aec
  autoencoder_inps = encoder_inps
  encoded = encoder(autoencoder_inps)
  decoded = decoder(encoded)

  autoencoder = models.Model(autoencoder_inps, decoded, name="autoencoder")

  return autoencoder


##### encoder as function ########
def build_encoder(input_shape, latent_dim, conv4):
    encoder_inputs = layers.Input(shape=input_shape, name="encoder_input")

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", strides=2)(encoder_inputs)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2D(conv4, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)

    x = layers.Flatten()(x)
    bottleneck = layers.Dense(latent_dim, activation="relu", name="bottleneck")(x)

    encoder = models.Model(encoder_inputs, bottleneck, name="encoder")
    encoder.summary()

    return encoder

##### decoder as function #####
def build_decoder(latent_dim, conv4):
    decoder_inputs = layers.Input(shape=(latent_dim,), name="decoder_input")

    x = layers.Dense(8 * 8 * latent_dim, activation="relu")(decoder_inputs)
    x = layers.Reshape((8, 8, latent_dim))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same")(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(conv4, (3, 3), activation="relu", padding="same")(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same")(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same", strides=2)(x)
    decoder_outputs = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same",
                                    name="decoder_output")(x)

    decoder = models.Model(decoder_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    return decoder


##### autoencoder as function (wiring encode + decode) #####
def build_autoencoder(input_shape, latent_dim, conv4):
    encoder = build_encoder(input_shape, latent_dim, conv4)
    decoder = build_decoder(latent_dim, conv4)

    autoencoder_inputs = layers.Input(shape=input_shape, name="ae_input")
    latent = encoder(autoencoder_inputs)
    reconstructed = decoder(latent)

    autoencoder = models.Model(autoencoder_inputs, reconstructed, name="autoencoder")
    autoencoder.summary()

    return autoencoder, encoder, decoder

# enc = build_encoder(input_shape=input_shape, latent_dim=latent_dim, conv4=conv4)
# print (enc)

# dec = build_decoder(latent_dim=latent_dim, conv4=conv4)
# print(dec)


aec, encoder, decoder = build_autoencoder(
    input_shape=input_shape,
    latent_dim=latent_dim,
    conv4=conv4
)


#aec = autoencoder(encoder_inps, latent_dim, conv4)
print(aec.summary())



### compile and callback
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

aec.compile(
    optimizer=optimizer,
    loss= "binary_crossentropy",
    metrics=["mse"])#MeanSquaredError())


checkpoint_path = base_path + f"./.weights_ld{latent_dim}conv4_{conv4}_lat_bkg.weights.h5"

lr_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [EarlyStopping(monitor="val_loss",
                           min_delta = 1e-3, # 5e-3,
                           patience=10,
                           restore_best_weights=True),
            ModelCheckpoint(filepath=checkpoint_path,
                             monitor="val_loss",
                             save_best_only=True,
                             save_weights_only=True), 
            WandbMetricsLogger(log_freq="epoch"),
            lr_plateau] #


#~~~~~~~~~~~
#train aec
#~~~~~~~~~~~~
history = aec.fit(bkg_train_norm_arr, bkg_train_norm_arr, 
                          epochs=200,
                          batch_size=64,
                          shuffle=True,
                          validation_split=0.1,
                          callbacks=callbacks)


train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
fig = plt.figure(figsize=(6,4))
plt.plot(range(len(train_loss)), train_loss)
plt.plot(range(len(val_loss)), val_loss)
plt.savefig(base_path + f"lat_bkg_loss_ld{latent_dim}conv4_{conv4}.png")
#plt.show()


### load model with best weights

aec.load_weights(checkpoint_path)


########################################
# Visualize original vs reconstructed
########################################

num_visualize = 5  # choose between 3–5
random_indices = np.random.choice(bkg_test_norm_arr.shape[0], num_visualize, replace=False)

sample_images = bkg_test_norm_arr[random_indices]

# reconstruct images
reconstructed_images = aec.predict(sample_images)

plt.figure(figsize=(10, 4))

for i in range(num_visualize):
    # original
    plt.subplot(2, num_visualize, i + 1)
    plt.imshow(sample_images[i])
    plt.title("Original")
    plt.axis("off")

    # reconstructed
    plt.subplot(2, num_visualize, i + 1 + num_visualize)
    plt.imshow(reconstructed_images[i])
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.savefig(base_path + f"reconstruction_ld{latent_dim}_conv4_{conv4}_lat_bkg.png")
# plt.show()




