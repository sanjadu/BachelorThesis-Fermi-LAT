import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import to_categorical

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap


import sklearn
import numpy
import matplotlib as mpl


#set correctly path for .keras
os.environ["HOME"] = "/d12/CAC/sd0017/"
os.environ["KERAS_HOME"] = "/d12/CAC/sd0017/downloads/.keras"   # change if needed


print("TF version:", tf.__version__)
print("sklearn version: ", sklearn.__version__)
print("matplotlib version: ", mpl.__version__)



#define laten dim and conv4
latent_dim = 256
conv4 = 64




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
src_list = []
for i in range(len(arr_list)):
    bkg = arr_list[i][0,:,:,:]
    bkg_list.append(bkg)
    src = arr_list[i][1,:,:,:]+arr_list[i][2,:,:,:]+arr_list[i][3,:,:,:]+arr_list[i][4,:,:,:]
    src_list.append(src)

bkg_list_to_arr = np.array(bkg_list)
print(bkg_list_to_arr.shape)
src_list_to_arr = np.array(src_list)

files_list_to_arr = np.array(arr_files)


(bkg_train, bkg_test, src_train, 
 src_test, files_train, files_test) = sklearn.model_selection.train_test_split(bkg_list_to_arr, src_list_to_arr, 
                                                                               files_list_to_arr, test_size=0.2, random_state=20)

print(bkg_train.shape)
print("tesst shape: ",bkg_test.shape)

bkg_train2 = bkg_train
bkg_test2 = bkg_test
src_test2 = src_test

tot_train = []
for x in zip(bkg_train, src_train):
    tot_train.append(x[0]+x[1])
tot_train = np.array(tot_train)

tot_test = []
for x in zip(bkg_test, src_test):
    tot_test.append(x[0]+x[1])
tot_test = np.array(tot_test)

tot_train2 = tot_train
tot_test2 = tot_test


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

src_test_norm = []
src_test2_norm = []
for i in range(len(src_test)):
    max_value_test = np.max(src_test[i])
    src_test_norm.append(src_test[i]/max_value_test)
    src_test2_norm.append(src_test2[i]/max_value_test)
src_test_norm_arr = np.array(src_test_norm)
src_test2_norm_arr = np.array(src_test2_norm)

tot_test_norm = []
tot_test2_norm = []
for i in range(len(tot_test)):
    max_value_test = np.max(tot_test[i])
    tot_test_norm.append(tot_test[i]/max_value_test)
    tot_test2_norm.append(tot_test2[i]/max_value_test)
tot_test_norm_arr = np.array(tot_test_norm)
tot_test2_norm_arr = np.array(tot_test2_norm)




###smaller batch fot test
#num_samples = 2000
#indices = np.random.choice(bkg_test_norm_arr.shape[0], num_samples, replace=False)

#bkg_test_subset = bkg_test_norm_arr[indices]
#bkg_test2_subset = bkg_test2_norm_arr[indices]



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
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

aec.compile(
    optimizer=optimizer,
    loss="binary_crossentropy")
    # should also test mse


checkpoint_path = base_path + f"./.weights_ld{latent_dim}conv4_{conv4}_lat_bkg.weights.h5"
#checkpoint_path = base_path + f"./.weights_ld{latent_dim}conv4_{conv4}_lat_src.weights.h5"



### load model with best weights

aec.load_weights(checkpoint_path)

## extract feats from bottleneck

# encoder contains bottleneck layer
z_test_bkg = encoder.predict(bkg_test_norm_arr, batch_size=256)
print("Bottleneck shape:", z_test_bkg.shape)

print(src_test_norm_arr.shape)

z_test_src = encoder.predict(src_test_norm_arr, batch_size=256)
print("Bottleneck shape:", z_test_src.shape)

print(src_test_norm_arr.shape)
print(tot_test_norm_arr.shape)

z_test_tot = encoder.predict(tot_test_norm_arr, batch_size=256)
print("Bottleneck shape:", z_test_tot.shape)



####----LATENT SPACE----#####

all_feats = np.concatenate([z_test_src, z_test_bkg, z_test_tot], axis=0)

###-PCA-###
pca = PCA(n_components=2)
reduced = pca.fit_transform(all_feats)

# Split reduced PCA coordinates
n_src = len(z_test_src)
n_bkg = len(z_test_bkg)
n_tot = len(z_test_tot)

src_pca = reduced[:n_src]
bkg_pca = reduced[n_src:n_src+n_bkg]
tot_pca = reduced[n_src+n_bkg:]

src_df = pd.DataFrame(src_pca, columns=["PC1", "PC2"])
bkg_df = pd.DataFrame(bkg_pca, columns=["PC1", "PC2"])
tot_df = pd.DataFrame(tot_pca, columns=["PC1", "PC2"])

# Optional: use separate colors to match scatter markers
palette = {"LAT Src": "green",
           "LAT Bkg": "gray",
           "LAT Tot": "blue", }


###-UMAP-###
min_dist = 0.3
reducer = umap.UMAP(
    n_neighbors=50,
    min_dist=min_dist,
    metric='euclidean'
)
umap_reduced = reducer.fit_transform(all_feats)

# Split UMAP coordinates (same logic as PCA)
src_umap = umap_reduced[:n_src]
bkg_umap = umap_reduced[n_src:n_src+n_bkg]
tot_umap = umap_reduced[n_src+n_bkg:]

src_umap_df = pd.DataFrame(src_umap, columns=["UMAP1", "UMAP2"])
bkg_umap_df = pd.DataFrame(bkg_umap, columns=["UMAP1", "UMAP2"])
tot_umap_df = pd.DataFrame(tot_umap, columns=["UMAP1", "UMAP2"])



###-PLOTTING-###
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

##--PCA PLOT(left)--##
ax0 = axes[0]
ax0.scatter(src_pca[:, 0], src_pca[:, 1], c='lightcoral', s=9, alpha=0.9, label='LAT Src', marker='*')
ax0.scatter(bkg_pca[:, 0], bkg_pca[:, 1], c='gray', s=10, alpha=0.5, label='LAT Bkg', marker='o')
ax0.scatter(tot_pca[:, 0], tot_pca[:, 1], c='mediumpurple', s=10, alpha=0.4, label='LAT Tot', marker='d')

# Add KDE contours
sns.kdeplot(data=src_df, x="PC1", y="PC2", ax=ax0, levels=5, color="lightcoral", linewidths=0.5, label=None)
sns.kdeplot(data=bkg_df, x="PC1", y="PC2", ax=ax0, levels=5, color="black", linewidths=0.5, label=None)
sns.kdeplot(data=tot_df, x="PC1", y="PC2", ax=ax0, levels=5, color="mediumpurple", linewidths=0.5, label=None)

ax0.set_title("PCA")
ax0.legend(fontsize=13)
ax0.tick_params(axis='both', labelsize=14)

##--UMAP PLOT(right)--##
ax1 = axes[1]

ax1.scatter(src_umap[:, 0], src_umap[:, 1], c='lightcoral', s=9, alpha=0.9, label='LAT Src', marker='*')
ax1.scatter(bkg_umap[:, 0], bkg_umap[:, 1], c='gray', s=10, alpha=0.5, label='LAT Bkg', marker='o')
ax1.scatter(tot_umap[:, 0], tot_umap[:, 1], c='mediumpurple', s=10, alpha=0.4, label='LAT Tot', marker='d')

sns.kdeplot(data=src_umap_df, x="UMAP1", y="UMAP2", ax=ax1, levels=5, color="lightcoral", linewidths=0.5)
sns.kdeplot(data=bkg_umap_df, x="UMAP1", y="UMAP2", ax=ax1, levels=5, color="black", linewidths=0.5)
sns.kdeplot(data=tot_umap_df, x="UMAP1", y="UMAP2", ax=ax1, levels=5, color="mediumpurple", linewidths=0.5)

ax1.set_title("UMAP")
ax1.legend(fontsize=13)
ax1.tick_params(axis='both', labelsize=14)


# Labels and save
plt.tight_layout()

plt.savefig(base_path + 'bkgw_tot_test_all_bottleneck_feats_PCA_countsColor_LAT_Tot_KDE_min_dist%s.png'%(min_dist), dpi=200)




###-PLOTTING ONLY PCA-###
fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))

##--PCA PLOT(left)--##
#ax0 = axes[0]
ax0.scatter(src_pca[:, 0], src_pca[:, 1], c='lightcoral', s=9, alpha=0.9, label='LAT Src', marker='*')
ax0.scatter(bkg_pca[:, 0], bkg_pca[:, 1], c='gray', s=10, alpha=0.5, label='LAT Bkg', marker='o')
ax0.scatter(tot_pca[:, 0], tot_pca[:, 1], c='mediumpurple', s=10, alpha=0.4, label='LAT Tot', marker='d')

# Add KDE contours
sns.kdeplot(data=src_df, x="PC1", y="PC2", ax=ax0, levels=5, color="lightcoral", linewidths=0.5, label=None)
sns.kdeplot(data=bkg_df, x="PC1", y="PC2", ax=ax0, levels=5, color="black", linewidths=0.5, label=None)
sns.kdeplot(data=tot_df, x="PC1", y="PC2", ax=ax0, levels=5, color="mediumpurple", linewidths=0.5, label=None)

ax0.set_title("PCA")
ax0.legend(fontsize=13)
ax0.tick_params(axis='both', labelsize=14)

plt.tight_layout()

plt.savefig(base_path + 'bkgw_tot_pca_latent_space', dpi=200)






########################################
# Visualize original vs reconstructed
########################################

num_visualize = 5  # choose between 3–5
random_indices = np.random.choice(tot_test_norm_arr.shape[0], num_visualize, replace=False)

sample_images = tot_test_norm_arr[random_indices]

# reconstruct images
reconstructed_images = aec.predict(sample_images)

plt.figure(figsize=(10, 4))

for i in range(num_visualize):
    # original
    plt.subplot(2, num_visualize, i + 1)
    plt.imshow(np.sqrt(sample_images[i]))
    plt.title("Original")
    plt.axis("off")

    # reconstructed
    plt.subplot(2, num_visualize, i + 1 + num_visualize)
    plt.imshow(np.sqrt(reconstructed_images[i]))
    plt.title("Reconstructed")
    plt.axis("off")

plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()
#plt.savefig(base_path + f"reconstruction_ld{latent_dim}_conv4_{conv4}_lat_src_bkgw.png", dpi=200)
#plt.savefig(base_path + f"reconstruction_ld{latent_dim}_conv4_{conv4}_lat_bkg.png", dpi=200)
plt.savefig(base_path + f"reconstruction_ld{latent_dim}_conv4_{conv4}_lat_tot_bkgw.png", dpi=200)