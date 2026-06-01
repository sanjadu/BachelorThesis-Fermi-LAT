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

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

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

(arr_list, num_list, arr_files) = datareader_bkg_src_Asi.read_npy_data(path_to_dir, channel=0, sqrt=0, train=True, norm=False)

dataset_test = []

for i in range(len(arr_files)):
    dataset_test.append({
        "id": arr_files[i],
        "bkg": arr_list[i][0,:,:,0],
        "src": arr_list[i][1,:,:,:]+arr_list[i][2,:,:,:]+arr_list[i][3,:,:,:]+arr_list[i][4,:,:,:]
    })


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

bkg_train2 = bkg_train
bkg_test2 = bkg_test
src_test2 = src_test

tot_all = []
for x in zip(bkg_list_to_arr, src_list_to_arr):
    tot_all.append(x[0]+x[1])
tot_all = np.array(tot_all)

# creating dataset in which images are connected with their name, all images
dataset_lat_all = []

for i in range(len(files_list_to_arr)):
    dataset_lat_all.append({
        "id": files_list_to_arr[i],
        "bkg": bkg_list_to_arr[i],
        "src": src_list_to_arr[i],
        "tot": tot_all[i]
    })



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

# creating dataset in which images are connected with their name, only test part of the images
dataset_lat = []

for i in range(len(files_test)):
    dataset_lat.append({
        "id": files_test[i],
        "bkg": bkg_test[i],
        "src": src_test[i],
        "tot": tot_test[i]
    })



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



# creating dataset in which images are connected with their name, test images that are already normalized
dataset_lat_norm = []

for i in range(len(files_test)):
    dataset_lat_norm.append({
        "id": files_test[i],
        "bkg": bkg_test_norm_arr[i],
        "src": src_test_norm_arr[i],
        "tot": tot_test_norm_arr[i]
    })





# check if correct name and image are connected
for i in range(len(dataset_lat_norm)):
    name = dataset_lat_norm[i]["id"]
    sum_bkg_lat = np.sum(dataset_lat_norm[i]["bkg"])
    
    found = False
    
    for j in range(len(dataset_test)):
        if name == dataset_test[j]["id"]:
            max = np.max(dataset_test[j]["bkg"])
            sum_bkg_test = np.sum(dataset_test[j]["bkg"]/max)
            #print("name lat",name)
            #print(dataset_lat_norm[i]["bkg"])
            #print("name test",dataset_test[j]["id"])
            #print(dataset_test[j]["bkg"])
            
            
            if np.isclose(sum_bkg_lat, sum_bkg_test):
                print(f"OK: {name}")
            else:
                print(f"Mismatch in BKG for {name}")
                print(f"  dataset_lat_norm sum: {sum_bkg_lat}")
                print(f"  dataset_test sum: {sum_bkg_test}")
            
            found = True
            break
    
    if not found:
        print(f"Name not found in dataset_test: {name}")



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



### load model with best weights

aec.load_weights(checkpoint_path)

## extract feats from bottleneck

# encoder contains bottleneck layer
z_test_bkg = encoder.predict(bkg_test_norm_arr, batch_size=256)
print("Bottleneck shape:", z_test_bkg.shape)

z_test_src = encoder.predict(src_test_norm_arr, batch_size=256)
print("Bottleneck shape:", z_test_src.shape)

z_test_tot = encoder.predict(tot_test_norm_arr, batch_size=256)
print("Bottleneck shape:", z_test_tot.shape)



np.random.seed(42)
indices = np.random.choice(len(z_test_bkg), size=100, replace=False)


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


bkg_pca_sub = bkg_pca[indices]
tot_pca_sub = tot_pca[indices]
src_pca_sub = src_pca[indices]

#getting the distance between two vectors in latent space (bkg,tot for same image)
pca_dist = np.linalg.norm(tot_pca_sub - bkg_pca_sub, axis=1) # shape (N,)
ids = [d["id"] for d in dataset_lat_norm]
ids_sub = [ids[i] for i in indices]

df_pairs = pd.DataFrame({
   #"file_b1": bkg_test_norm_arr,
   #"file_b2": tot_test_norm_arr,
   "id": ids_sub,
   "PCAdist": pca_dist
})
#plotting histogram
plt.hist(pca_dist, bins=12)
plt.xlabel("PCA Distance")
plt.ylabel("Frequency")
plt.title("Distribution of Latent Space Distances (PCA)")
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()

plt.savefig(base_path + f'distances_hist_pca_lat_ld{latent_dim}conv4_{conv4}.png', dpi=200)

#getting top 5% and bottom 5% of the images on the histogram
low_thresh = np.percentile(pca_dist, 5)
high_thresh = np.percentile(pca_dist, 95)
low_5 = df_pairs[df_pairs["PCAdist"] <= low_thresh]
high_5 = df_pairs[df_pairs["PCAdist"] >= high_thresh]

dataset_dict = {d["id"]: d for d in dataset_lat}

low_ids = low_5["id"].values
high_ids = high_5["id"].values

# IDs that should have connecting lines
selected_ids = set(low_ids).union(set(high_ids))

low_images = [dataset_dict[i] for i in low_ids]
high_images = [dataset_dict[i] for i in high_ids]

dist_dict = dict(zip(df_pairs["id"], df_pairs["PCAdist"]))

for d in low_images + high_images:
    d["PCAdist"] = dist_dict[d["id"]]


def show_grid_low(samples, n=5):
    # sort by PCA distance
    samples = sorted(samples, key=lambda x: x["PCAdist"])

    n = min(n, len(samples))  # safety

    fig, axes = plt.subplots(n, 3, figsize=(9, 3*n))

    # If only 1 sample, axes is 1D → fix shape
    if n == 1:
        axes = [axes]

    for i in range(n):
        d = samples[i]

        im_bkg = axes[i][0].imshow(d["bkg"])
        axes[i][0].set_title("bkg", fontsize=13)
        fig.colorbar(im_bkg, ax=axes[i][0], fraction=0.046, pad=0.04)

        im_src = axes[i][1].imshow(d["src"])
        axes[i][1].set_title("src", fontsize=13)
        fig.colorbar(im_src, ax=axes[i][1], fraction=0.046, pad=0.04)

        im_tot = axes[i][2].imshow(d["tot"])
        axes[i][2].set_title(f"tot\nDist={d['PCAdist']:.3f}", fontsize=13)
        fig.colorbar(im_tot, ax=axes[i][2], fraction=0.046, pad=0.04)

        # optional: show ID on the leftž
        axes[i][0].set_ylabel(d["id"], rotation=0, labelpad=40, fontsize=8)

        # remove ticks
        for j in range(3):
            axes[i][j].axis("off")

    plt.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    plt.savefig(base_path + f'low5_pca_lat_ld{latent_dim}conv4_{conv4}.png', dpi=200)
    #plt.show()


def show_grid_high(samples, n=5):
    # sort by PCA distance
    samples = sorted(samples, key=lambda x: x["PCAdist"], reverse=True)

    n = min(n, len(samples))  # safety

    fig, axes = plt.subplots(n, 3, figsize=(9, 3*n))

    # If only 1 sample, axes is 1D → fix shape
    if n == 1:
        axes = [axes]

    for i in range(n):
        d = samples[i]

        im_bkg = axes[i][0].imshow(d["bkg"])
        axes[i][0].set_title("bkg", fontsize=13)
        fig.colorbar(im_bkg, ax=axes[i][0], fraction=0.046, pad=0.04)

        im_src = axes[i][1].imshow(d["src"])
        axes[i][1].set_title("src", fontsize=13)
        fig.colorbar(im_src, ax=axes[i][1], fraction=0.046, pad=0.04)

        im_tot = axes[i][2].imshow(d["tot"])
        axes[i][2].set_title(f"tot\nDist={d['PCAdist']:.3f}", fontsize=13)
        fig.colorbar(im_tot, ax=axes[i][2], fraction=0.046, pad=0.04)

        # optional: show ID on the left
        axes[i][0].set_ylabel(d["id"], rotation=0, labelpad=40, fontsize=8)

        # remove ticks
        for j in range(3):
            axes[i][j].axis("off")

    plt.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    plt.savefig(base_path + f'top5_pca_lat_ld{latent_dim}conv4_{conv4}.png', dpi=200)
    #plt.show()

print("LOW 5%")
show_grid_low(low_images, n=5)

print("HIGH 5%")
show_grid_high(high_images, n=5)


###-UMAP-###
reducer = umap.UMAP(
    n_neighbors=50,
    min_dist=0.1,
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

bkg_umap_sub = bkg_umap[indices]
tot_umap_sub = tot_umap[indices]
src_umap_sub = src_umap[indices]

#getting the distance between two vectors in latent space (bkg,tot for same image)
umap_dist = np.linalg.norm(tot_umap_sub - bkg_umap_sub, axis=1) # shape (N,)
ids = [d["id"] for d in dataset_lat]
ids_sub = [ids[i] for i in indices]

df_pairs = pd.DataFrame({
   #"file_b1": bkg_test_norm_arr,
   #"file_b2": tot_test_norm_arr,
   "id": ids_sub,
   "UMAPdist": umap_dist
})
#plotting histogram
plt.hist(umap_dist, bins=20)
plt.xlabel("UMAP Distance")
plt.ylabel("Frequency")
plt.title("Distribution of Latent Space Distances (UMAP)")
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()

plt.savefig(base_path + f'distances_hist_umap_lat_ld{latent_dim}conv4_{conv4}.png', dpi=200)



###-PLOTTING-###
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

##--PCA PLOT(left)--##
ax0 = axes[0]
ax0.scatter(src_pca_sub[:, 0], src_pca_sub[:, 1], c='lightcoral', s=20, alpha=0.9, label='LAT Src', marker='*')
ax0.scatter(bkg_pca_sub[:, 0], bkg_pca_sub[:, 1], c='gray', s=30, alpha=0.9, label='LAT Bkg', marker='o')
ax0.scatter(tot_pca_sub[:, 0], tot_pca_sub[:, 1], c='mediumpurple', s=30, alpha=0.9, label='LAT Tot', marker='d')

for i in range(len(indices)):

    # only connect selected images
    if ids_sub[i] not in selected_ids:
        continue

    ax0.plot(
        [bkg_pca_sub[i, 0], tot_pca_sub[i, 0]],
        [bkg_pca_sub[i, 1], tot_pca_sub[i, 1]],
        color='black',
        alpha=0.8,
        linewidth=1.5
    )

ax0.set_title("PCA")
ax0.legend(fontsize=13)
ax0.tick_params(axis='both', labelsize=14)

##--UMAP PLOT(right)--##
ax = axes[1]

ax.scatter(src_umap_sub [:, 0], src_umap_sub [:, 1], c='lightcoral', s=9, alpha=0.9, label='LAT Src', marker='*')
ax.scatter(bkg_umap_sub[:, 0], bkg_umap_sub[:, 1], c='gray', s=10, alpha=0.9, label='LAT Bkg', marker='o')
ax.scatter(tot_umap_sub[:, 0], tot_umap_sub[:, 1], c='mediumpurple', s=10, alpha=0.9, label='LAT Tot', marker='d')

for i in range(len(indices)):

    # only connect selected images
    if ids_sub[i] not in selected_ids:
        continue

    ax.plot(
        [bkg_umap_sub[i, 0], tot_umap_sub[i, 0]],
        [bkg_umap_sub[i, 1], tot_umap_sub[i, 1]],
        color='black',
        alpha=0.8,
        linewidth=1.5
    )

ax.set_title("UMAP")
ax.legend(fontsize=13)


plt.legend(fontsize=11)
# Labels and save
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()

plt.savefig(base_path + 'bkgw_latent_space_connected.png', dpi=200)





###-PLOTTING ONLY PCA-###
fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))

ax0.scatter(src_pca_sub[:, 0], src_pca_sub[:, 1], c='lightcoral', s=20, alpha=0.9, label='LAT Src', marker='*')
ax0.scatter(bkg_pca_sub[:, 0], bkg_pca_sub[:, 1], c='gray', s=30, alpha=0.9, label='LAT Bkg', marker='o')
ax0.scatter(tot_pca_sub[:, 0], tot_pca_sub[:, 1], c='mediumpurple', s=30, alpha=0.9, label='LAT Tot', marker='d')

for i in range(len(indices)):

    # only connect selected images
    if ids_sub[i] not in selected_ids:
        continue

    ax0.plot(
        [bkg_pca_sub[i, 0], tot_pca_sub[i, 0]],
        [bkg_pca_sub[i, 1], tot_pca_sub[i, 1]],
        color='black',
        alpha=0.8,
        linewidth=1.5
    )

ax0.set_title("PCA")
ax0.legend(fontsize=13)
ax0.tick_params(axis='both', labelsize=14)


# Labels and save
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()

plt.savefig(base_path + 'bkgw_latent_space_connected_pcaonly.png', dpi=200)








#### plotting two images ####
print(type(dataset_lat_all))
for img in dataset_lat_all:
    #print(img.get("id"))
    if str(img["id"]) == "test_image_691200.npy":
        print("je not")
        bkg_out = img.get("bkg")
        src_out = img.get("src")
        tot_out = img.get("tot")
    elif img.get("id") == "test_image_691507.npy":
        bkg_in = img.get("bkg")
        src_in = img.get("src")
        tot_in = img.get("tot")
    

fig, axes = plt.subplots(2, 3, figsize=(9, 6))
## out of galactic plane
fig.text(0.5, 0.94, "Out of galactic plane", ha="center", fontsize=14, fontweight="bold")
im_bkg = axes[0][0].imshow(np.sqrt(bkg_out))
axes[0][0].set_title("bkg", fontsize=13)
axes[0][0].axis("off")
fig.colorbar(im_bkg, ax=axes[0][0], fraction=0.046, pad=0.04)

im_src = axes[0][1].imshow(np.sqrt(src_out))
axes[0][1].set_title("src", fontsize=13)
axes[0][1].axis("off")
fig.colorbar(im_src, ax=axes[0][1], fraction=0.046, pad=0.04)

im_tot = axes[0][2].imshow(np.sqrt(tot_out))
axes[0][2].set_title("tot", fontsize=13)
axes[0][2].axis("off")
cbar = fig.colorbar(im_tot, ax=axes[0][2], fraction=0.046, pad=0.04)
cbar.set_label(r"$\sqrt{N}$")



## in the galactic plane
fig.text(0.5, 0.43, "Insied the galactic plane", ha="center", fontsize=14, fontweight="bold")
im_bkg = axes[1][0].imshow(np.sqrt(bkg_in))
axes[1][0].set_title("bkg", fontsize=13)
axes[1][0].axis("off")
fig.colorbar(im_bkg, ax=axes[1][0], fraction=0.046, pad=0.04)

im_src = axes[1][1].imshow(np.sqrt(src_in))
axes[1][1].set_title("src", fontsize=13)
axes[1][1].axis("off")
fig.colorbar(im_src, ax=axes[1][1], fraction=0.046, pad=0.04)

im_tot = axes[1][2].imshow(np.sqrt(tot_in))
axes[1][2].set_title("tot", fontsize=13)
axes[1][2].axis("off")
cbar = fig.colorbar(im_tot, ax=axes[1][2], fraction=0.046, pad=0.04)
cbar.set_label(r"$\sqrt{N}$", fontsize=13)


plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.88, hspace=0.45)
plt.savefig(base_path + 'img_lat_in_out_galactic_plane.png', dpi=200)