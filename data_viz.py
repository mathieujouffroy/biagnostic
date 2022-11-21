import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical


# Select random layer number
def explore_3dimage(image, layer, channel):
    plt.figure(figsize=(8, 5))
    #channel = 3
    plt.imshow(image[:, :, layer, channel], cmap='gray')
    plt.title('Explore Layers of Brain MRI', fontsize=10)
    plt.axis('off')
    return layer, channel


def explore_label_per_channel(label, layer):
    classes_dict = {
        'Normal': 0.,
        'Edema': 1.,
        'Non-enhancing tumor': 2.,
        'Enhancing tumor': 3.
    }
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 20))
    for i in range(4):
        label_str = list(classes_dict.keys())[i]
        img = label[:,:,layer]
        mask = np.where(img == classes_dict[label_str], 255, 0)
        ax[i].imshow(mask)
        ax[i].set_title(f"{label_str}", fontsize=10)
        ax[i].axis('off')

    return layer


def explore_label(label, layer):
    classes_dict = {
        'Normal': 0.,
        'Edema': 1.,
        'Non-enhancing tumor': 2.,
        'Enhancing tumor': 3.
    }
    plt.figure(figsize=(8, 5))
    #channel = 3
    plt.imshow(label[:, :, layer])
    plt.title('Explore Layers of Brain MRI', fontsize=10)
    plt.axis('off')
    return layer


def colorize_labels_image_flair(image, label, is_categorical=False):
    if not is_categorical:
        # one hot encode labels
        label = to_categorical(label, num_classes=4).astype(np.uint8)

    # Normalize Flair channel
    image = cv2.normalize(image[:, :, :, 0], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

    un, cn = np.unique(image, return_counts=True)
    #print(np.asarray((un, cn)).T)

    # create array without taking into account the background label
    labeled_image = np.zeros_like(label[:, :, :, 1:])

    # remove tumor part from image
    # (remember we removed the background, so index 0 -> 'edema')
    # label[:, :, :, 0] -> background, all 1 for background
    labeled_image[:, :, :, 0] = image * (label[:, :, :, 0])
    labeled_image[:, :, :, 1] = image * (label[:, :, :, 0])
    labeled_image[:, :, :, 2] = image * (label[:, :, :, 0])

    labeled_image += (label[:, :, :, 1:] * 255)

    return labeled_image


def plot_image_grid(image):
    data_all = []
    data_all.append(image)

    fig, ax = plt.subplots(3, 6, figsize=[16, 9])

    # coronal plane
    # transpose : batch, h, w, depth, channel -> height, depth, width, channel, batch
    coronal = np.transpose(data_all, [1, 3, 2, 4, 0])
    coronal = np.rot90(coronal, k=1)

    # transversal plane
    # transpose : batch, h, w, depth, channel -> width, height, depth, channel, batch
    transversal = np.transpose(data_all, [2, 1, 3, 4, 0])
    transversal = np.rot90(transversal, k=2)

    # sagittal plane
    # transpose : batch, h, w, depth, channel -> width, depth, height, channel, batch
    sagittal = np.transpose(data_all, [2, 3, 1, 4, 0])
    sagittal = np.rot90(sagittal, k=1)

    for i in range(6):
        n = np.random.randint(40, coronal.shape[2]-40)
        ax[0][i].imshow(np.squeeze(coronal[:, :, n, :]))
        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])
        if i == 0:
            ax[0][i].set_ylabel('Coronal', fontsize=15)

    for i in range(6):
        n = np.random.randint(40, transversal.shape[2]-40)
        ax[1][i].imshow(np.squeeze(transversal[:, :, n, :]))
        ax[1][i].set_xticks([])
        ax[1][i].set_yticks([])
        if i == 0:
            ax[1][i].set_ylabel('Transversal', fontsize=15)

    for i in range(6):
        n = np.random.randint(40, sagittal.shape[2]-40)
        ax[2][i].imshow(np.squeeze(sagittal[:, :, n, :]))
        ax[2][i].set_xticks([])
        ax[2][i].set_yticks([])
        if i == 0:
            ax[2][i].set_ylabel('Sagittal', fontsize=15)

    fig.subplots_adjust(wspace=0, hspace=0)


def visualize_data_gif(data_):
    images = []
    for i in range(data_.shape[0]):
        x = data_[min(i, data_.shape[0] - 1), :, :]
        y = data_[:, min(i, data_.shape[1] - 1), :]
        z = data_[:, :, min(i, data_.shape[2] - 1)]
        img = np.concatenate((x, y, z), axis=1)
        images.append(img)
    imageio.mimsave("/tmp/gif.gif", images, duration=0.01)
    return Image(filename="/tmp/gif.gif", format='png')
