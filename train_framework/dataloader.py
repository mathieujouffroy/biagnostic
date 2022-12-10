import os
import json
import cv2
import h5py
import torch
import logging
import numpy as np
import nilearn as nl
import nibabel as nib
import tensorflow as tf
import multiprocessing
import matplotlib.pyplot as plt
import nilearn.plotting as nlplt
import monai.transforms as transforms

logger = logging.getLogger(__name__)

def store_hdf5(name, images, masks):
    """
    Stores an array of images to HDF5.

    Args:
        name(str):				filename
        images(numpy.array):    images array
        masks(numpy.array): 	segmentation mask array

    Returns:
        file(h5py.File): file containing
    """

    # Create a new HDF5 file
    file = h5py.File(name, "w")
    
    # Images are store as float 32 & Masks as uint8 
    file.create_dataset("images", np.shape(images), data=images)
    file.create_dataset("masks", np.shape(masks), h5py.h5t.STD_U8BE, data=masks)
    file.close()
    return file


class BratsDatasetGenerator:

    def __init__(self, args):

        self.ds_path = args.ds_path
        self.n_channels = args.n_channels
        self.batch_size = args.batch_size
        self.crop_shape = args.crop_shape
        self.n_classes = args.n_classes
        self.crop_dim = (args.crop_shape[0], args.crop_shape[1], args.crop_shape[2], args.n_channels)
        self.train_val_split = args.train_val_split
        self.val_test_split = args.val_test_split
        self.seed = args.seed
        self.create_file_list()
        self.ds_train, self.ds_val, self.ds_test = self.get_dataset()
        #print("got ds")

    def create_file_list(self):
        """
        Get list of the files from the BraTS raw data and create
        a dictionary of tuples with image filename and label filename.
        """

        json_filename = os.path.join(self.ds_path, "dataset.json")

        try:
            with open(json_filename, "r") as fp:
                experiment_data = json.load(fp)
        except IOError as e:
            print(f"File {json_filename} doesn't exist. It should be part of the "
                  "Decathlon directory")

        self.output_channels = experiment_data["labels"]
        self.input_channels = experiment_data["modality"]
        self.description = experiment_data["description"]
        self.name = experiment_data["name"]
        self.release = experiment_data["release"]
        self.reference = experiment_data["reference"]
        self.tensorImageSize = experiment_data["tensorImageSize"]
        self.numFiles = experiment_data["numTraining"]
        #self.numFiles = 8

        self.len_train = int(self.numFiles * self.train_val_split)
        self.val_test_len = self.numFiles - self.len_train
        self.len_val = int(self.val_test_len * self.val_test_split)
        self.len_test = self.numFiles - self.len_train - self.len_val
        self.train_ids = [i for i in range(self.len_train)]
        self.val_ids = [i for i in range(self.len_train, self.len_train+self.len_val)]
        self.test_ids = [i for i in range(self.len_train+self.len_val, self.len_train+self.len_val+self.len_test)]

        self.filenames = {}
        for idx in range(self.numFiles):

            img_path = experiment_data["training"][idx]["image"][2:]
            label_path = experiment_data["training"][idx]["label"][2:]

            self.filenames[idx] = [os.path.join(self.ds_path,img_path),
                                    os.path.join(self.ds_path,label_path)]

            if idx == 0:
                self.img_shape = np.array(nib.load(self.filenames[idx][0]).get_fdata()).shape
                self.label_shape = np.array(nib.load(self.filenames[idx][1]).get_fdata()).shape


    def print_info(self, log=False):
        """ Print the dataset information """

        if log:
            logger.info("="*40)
            logger.info(f"Dataset name:        {self.name}")
            logger.info(f"Dataset description: {self.description}")
            logger.info(f"Dataset release:     {self.release}")
            logger.info(f"Dataset reference:   {self.reference}")
            logger.info(f"Input channels:      {self.input_channels}")
            logger.info(f"Output labels:       {self.output_channels}")
            logger.info(f"Tensor image size:   {self.tensorImageSize}")
            logger.info(f"Nbr of images:       {self.numFiles}")
            logger.info(f"Image shape:         {self.img_shape}")
            logger.info(f"Label shape:         {self.label_shape}")
            logger.info(f"Training set len:    {self.len_train}")
            logger.info(f"Validation set len:  {self.len_val}")
            logger.info(f"Testing set len:     {self.len_test}")
            logger.info("="*40)
        else:
            print("="*40)
            print(f"Dataset name:        {self.name}")
            print(f"Dataset description: {self.description}")
            print(f"Dataset release:     {self.release}")
            print(f"Dataset reference:   {self.reference}")
            print(f"Input channels:      {self.input_channels}")
            print(f"Output labels:       {self.output_channels}")
            print(f"Tensor image size:   {self.tensorImageSize}")
            print(f"Nbr of images:       {self.numFiles}")
            print(f"Image shape:         {self.img_shape}")
            print(f"Label shape:         {self.label_shape}")
            print(f"Training set len:    {self.len_train}")
            print(f"Validation set len:  {self.len_val}")
            print(f"Testing set len:     {self.len_test}")
            print("="*40)


    def load_example(self, idx):
        # load the image and label file, get the image content and return a numpy array for each
        image = np.array(nib.load(self.filenames[idx][0]).get_fdata())
        label = np.array(nib.load(self.filenames[idx][1]).get_fdata())
        return image, label


    def get_subvol_coords_lowest_bgd(self, idx, max_tries = 150, verbose=False):
        """
        Extract random sub-volume from original images.

        Args:
            max_tries (int): maximum trials to do when sampling
            background_threshold (float): limit on the fraction
                of the sample which can be the background

        returns:
            X (np.array): sample of original image of dimension
                (n_channels, output_x, output_y, output_z)
            y (np.array): labels which correspond to X, of dimension
                (n_classes, output_x, output_y, output_z)
        """

        orig_x, orig_y, orig_z = self.img_shape[0], self.img_shape[1], self.img_shape[2]
        output_x, output_y, output_z = self.crop_shape[0], self.crop_shape[1], self.crop_shape[2]

        _, label = self.load_example(idx)
        tries = 0

        #min_x = min(np.where(label != 0)[0])
        #max_x = max(np.where(label != 0)[0])
        #min_y = min(np.where(label != 0)[1])
        #max_y = max(np.where(label != 0)[1])
        min_x = min(np.where(label != 0)[1])
        max_x = max(np.where(label != 0)[1])
        min_y = min(np.where(label != 0)[0])
        max_y = max(np.where(label != 0)[0])

        min_z = min(np.where(label != 0)[2])
        max_z = max(np.where(label != 0)[2])

        start_x = (orig_x - output_x + 1)//2
        start_y = (orig_y - output_y + 1)//2

        if start_x > min_x - 5:
            start_x = min_x - 5
        if start_y > min_y - 5:
            start_y = min_y - 5

        if start_x + output_x <= max_x + 1:
            start_x = max_x - output_x + 1
        if start_y + output_y <= max_y + 1:
            start_y = max_y - output_y + 1

        best_bgrd_ratio = 1

        if (max_z - min_z <= 64):
            best_z = min_z
            if best_z + output_z >= orig_z:
                best_z = orig_z - output_z

            y = label[start_x: start_x + output_x,
                      start_y: start_y + output_y,
                      best_z: best_z + output_z]
            y = np.eye(self.n_classes, dtype='uint8')[y.astype(int)]
            bgrd_ratio = np.sum(y[:,:,:,0]) / (output_x * output_y * output_z)
            if verbose:
                print(f"\nRatio: {best_bgrd_ratio}")
                print(f"start_X : {start_x}, end_X: {start_x+output_x}, min_label_x: {min_x}, max_label_x: {max_x}, len_x = {max_x-min_x}")
                print(f"start_Y : {start_y}, end_X: {start_y+output_y}, min_label_y: {min_y}, max_label_y: {max_y}, len_y = {max_y-min_y}")
                print(f"Start_Z: {best_z}, end_Z: {best_z+output_z}, min_label_z: {min_z}, max_label_z: {max_z}, len_z: {max_z-min_z}")
            return int(start_x), int(start_x), int(best_z), bgrd_ratio

        while tries < max_tries:
            start_z = np.random.randint(min_z-1, max_z-output_z)
            y = label[start_x: start_x + output_x,
                      start_y: start_y + output_y,
                      start_z: start_z + output_z]

            # One-hot encode the categories -> (output_x, output_y, output_z, n_classes)
            y = np.eye(self.n_classes, dtype='uint8')[y.astype(int)]

            # compute the background ratio
            bgrd_ratio = np.sum(y[:,:,:,0]) / (output_x * output_y * output_z)

            # keep z coordinates with the smallest background ratio
            if bgrd_ratio < best_bgrd_ratio:
                best_bgrd_ratio = bgrd_ratio
                best_z = start_z

            tries += 1

        if verbose:
            print(f"\nRatio: {best_bgrd_ratio}")
            print(f"start_X : {start_x}, min_label_x: {min_x}, max_label_x: {max_x}, end_X: {start_x+output_x}, len_x = {max_x-min_x}")
            print(f"start_Y : {start_y}, min_label_y: {min_y}, max_label_y: {max_y}, end_Y: {start_y+output_y}, len_y = {max_y-min_y}")
            print(f"Start_Z: {best_z}, min_label_z: {min_z}, max_label_z: {max_z}, end_Z: {best_z+output_z}, len_z: {max_z-min_z}")
        return int(start_x), int(start_y), int(best_z), best_bgrd_ratio


    def gen_subvol_coords(self):
        coord_dict = dict()

        for i in range(self.numFiles):
            start_x, start_y, start_z, ratio = self.get_subvol_coords_lowest_bgd(i)
            coord_dict[i] = {"start_x":start_x, "start_y":start_y, "start_z":start_z, "ratio": ratio}

        with open(f'{self.ds_path}volumes_coord.json', 'w') as f:
            json.dump(coord_dict, f, indent=4)


    def get_subvol_coords(self, idx):
        with open(f'{self.ds_path}volumes_coord.json', "r")  as f:
            id_coords_dict = json.load(f)

        coords_dict = id_coords_dict[str(idx)]
        #start_x = coords_dict["start_x"]
        #start_y = coords_dict["start_y"]
        start_x = coords_dict["start_y"]
        start_y = coords_dict["start_x"]
        start_z = coords_dict["start_z"]

        return start_x, start_y, start_z


    def standardize(self, image):
        """
        Standardize mean and standard deviation of each channel and z_dimension.

        Args:
            image (np.array): input image, shape (n_channels, dim_x, dim_y, dim_z)

        Returns:
            standardized_image (np.array): standardized version of input image
        """

        standardized_image = np.zeros(image.shape)

        # iterate over channels
        for c in range(image.shape[0]):
            # iterate over the `depth` dimension
            for z in range(image.shape[3]):
                image_slice = image[c, :, : ,z]
                centered = image_slice - np.mean(image_slice)
                if np.std(centered) == 0:
                    centered_scaled = centered
                else:
                    centered_scaled = centered / np.std(centered)
                standardized_image[c, :, :, z] = centered_scaled

        return standardized_image


    def generate_sub_volume(self, idx, store=True):
        """
        Generate sub-volume from original images.

        Args:
            max_tries (int): maximum trials to do when sampling
            background_threshold (float): limit on the fraction
                of the sample which can be the background

        returns:
            X (np.array): sample of original image of dimension
                (n_channels, output_x, output_y, output_z)
            y (np.array): labels which correspond to X, of dimension
                (n_classes, output_x, output_y, output_z)
        """
        output_x, output_y, output_z = self.crop_shape[0], self.crop_shape[1], self.crop_shape[2]

        #idx = idx.numpy()
        image, label = self.load_example(idx)
        start_x, start_y, start_z = self.get_subvol_coords(idx)

        y = label[start_x: start_x + output_x,
                  start_y: start_y + output_y,
                  start_z: start_z + output_z]
        # One-hot encode the categories -> (output_x, output_y, output_z, n_classes)
        y = np.eye(self.n_classes, dtype='uint8')[y.astype(int)]
        X = np.copy(image[start_x: start_x + output_x,
                          start_y: start_y + output_y,
                          start_z: start_z + output_z, :])

        # (x_dim, y_dim, z_dim, n_channels) -> (n_channels, x_dim, y_dim, z_dim)
        X = np.moveaxis(X,3,0)
        # (x_dim, y_dim, z_dim, n_classes) -> (n_classes, x_dim, y_dim, z_dim)
        y = np.moveaxis(y,3,0)

        assert X.shape[1] == y.shape[1] == output_x
        assert X.shape[2] == y.shape[2] == output_y
        assert X.shape[3] == y.shape[3] == output_z

        # exclude background class in the 'n_classes' dimension
        mask = y[1:, :, :, :]

        image = self.standardize(X)

        if not store:
            return image, y
        else:
            name = f"subvolumes/BRATS_{idx}_{start_x}_{start_y}_{start_z}.h5"
            path_name = os.path.join(self.ds_path, name)
            store_hdf5(path_name, image, mask)
            return


    def get_sub_volume(self, idx):
        img, label = self.load_example(idx)
        start_x, start_y, start_z = self.get_subvol_coords(idx)

        img = img[start_x: start_x+self.crop_shape[0],
                start_y: start_y+self.crop_shape[1],
                start_z: start_z+self.crop_shape[2], :]

        mask = np.eye(self.n_classes, dtype='uint8')[label.astype(int)]

        mask = mask[start_x: start_x + self.crop_shape[0],
                    start_y: start_y + self.crop_shape[1],
                    start_z: start_z + self.crop_shape[2]]

        # change dimension from (x_dim, y_dim, z_dim, n_channels)  to (n_channels, x_dim, y_dim, z_dim)
        img = np.moveaxis(img,3,0)
        # change dimension from (x_dim, y_dim, z_dim, n_classes) to (n_classes, x_dim, y_dim, z_dim)
        mask = np.moveaxis(mask,3,0)
        img = self.standardize(img)
        print(f"start_x: {start_x}, start_y:{start_y}, start_z:{start_z}")
        return img, mask


    def plot_example(self, idx, depth):
        img, label = self.load_example(idx)
        img = img[:, :, :, 0]
        plt.figure("image with mask", (18, 6))
        plt.subplot(1, 2, 1)
        plt.title("image")
        plt.imshow(img[:, :, depth], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title("tumor mask")
        plt.imshow(label[:, :, depth])
        plt.show()


    def plot_nl(self, idx):
        niimg = nl.image.load_img(self.filenames[idx][0])
        nimask = nl.image.load_img(self.filenames[idx][1])
        niimg = niimg.slicer[:, :, :, 0]
        fig, axes = plt.subplots(nrows=4, figsize=(8, 10))
        nlplt.plot_img(niimg, axes=axes[0])
        nlplt.plot_anat(niimg, axes=axes[1])
        nlplt.plot_epi(niimg, axes=axes[2])
        nlplt.plot_roi(nimask, bg_img=niimg,  axes=axes[3], cmap='Paired')
        plt.show()


    def explore_3D_image(self, id, layer, channel):
        classes_dict = {
            'Normal': 0.,
            'Edema': 1.,
            'Non-enhancing tumor': 2.,
            'Enhancing tumor': 3.
        }
        image, label = self.load_example(id)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
        ax[0].imshow(image[:, :, layer, channel], cmap='gray')
        ax[1].imshow(label[:, :, layer])
        fig.suptitle('Explore Layers of Brain MRI', fontsize=10)

        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 20))
        for i in range(4):
            label_str = list(classes_dict.keys())[i]
            img = label[:,:,layer]
            mask = np.where(img == classes_dict[label_str], 255, 0)
            ax[i].imshow(mask)
            ax[i].set_title(f"{label_str}", fontsize=10)
            ax[i].axis('off')


    def colorize_labels_image_flair(self, image, label):
        label = np.eye(self.n_classes, dtype="uint8")[label.astype(int)]
        ## Normalize Flair channel
        #image = image[:, :, :, 0]
        image = cv2.normalize(image[:, :, :, 0], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

        # create array without taking into account the background label
        labeled_image = np.zeros_like(label[:, :, :, 1:])

        # remove tumor part from image
        # (remember we removed the background, so index 0 -> 'edema')
        # label[:, :, :, 0] -> all 1 for background, all 0 for the tumor parts
        labeled_image[:, :, :, 0] = image * (label[:, :, :, 0])
        labeled_image[:, :, :, 1] = image * (label[:, :, :, 0])
        labeled_image[:, :, :, 2] = image * (label[:, :, :, 0])

        labeled_image += (label[:, :, :, 1:] * 255)
        return labeled_image


    def plot_image_grid(self, image, label):
        image = self.colorize_labels_image_flair(image, label)

        data_all = []
        data_all.append(image)

        fig, ax = plt.subplots(3, 6, figsize=[16, 9])

        # coronal plane
        # batch, h, w, depth, channel -> height, depth, width, channel, batch
        coronal = np.transpose(data_all, [1, 3, 2, 4, 0])
        coronal = np.rot90(coronal, k=1)

        # transversal plane
        # batch, h, w, depth, channel -> width, height, depth, channel, batch
        transversal = np.transpose(data_all, [2, 1, 3, 4, 0])
        transversal = np.rot90(transversal, k=2)

        # sagittal plane
        # batch, h, w, depth, channel -> width, depth, height, channel, batch
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


    def visualize_patch(self, depth, img_chan, image, mask, title=None):
        plt.figure("image", (8, 6))
        plt.subplot(1, 2, 1)
        plt.title('MRI Flair Channel')
        plt.imshow(image[img_chan, :, :, depth], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title('Mask')
        plt.imshow(mask[:, :, depth])
        if title:
            plt.suptitle(title)
        plt.show()


    def visualize_patch_pred(self, idx, img_chan, image, mask, pred_mask):
        self.visualize_patch(idx, img_chan, image, mask, "true mask")
        self.visualize_patch(idx, img_chan, image, pred_mask, "pred mask")


    def predict_and_viz(self, image, label, model, threshold, loc=(100, 100, 50)):
        image_labeled = self.colorize_labels_image_flair(image.copy(), label.copy())

        model_label = np.zeros([3, 320, 320, 160])

        for x in range(0, image.shape[0], 160):
            for y in range(0, image.shape[1], 160):
                for z in range(0, image.shape[2], 16):
                    patch = np.zeros([4, 160, 160, 16])
                    p = np.moveaxis(image[x: x + 160, y: y + 160, z:z + 16], 3, 0)
                    patch[:, 0:p.shape[1], 0:p.shape[2], 0:p.shape[3]] = p
                    pred = model.predict(np.expand_dims(patch, 0))
                    model_label[:, x:x + p.shape[1],
                    y:y + p.shape[2],
                    z: z + p.shape[3]] += pred[0][:, :p.shape[1], :p.shape[2],
                                          :p.shape[3]]

        model_label = np.moveaxis(model_label[:, 0:240, 0:240, 0:155], 0, 3)
        model_label_reformatted = np.zeros((240, 240, 155, 4))

        model_label_reformatted = np.eye(self.n_classes, dtype="uint8")(y.astype(int))


        model_label_reformatted[:, :, :, 1:4] = model_label

        model_labeled_image = self.colorize_labels_image_flair(image, model_label_reformatted,
                                                is_categorical=True)

        fig, ax = plt.subplots(2, 3, figsize=[10, 7])

        # plane values
        x, y, z = loc

        ax[0][0].imshow(np.rot90(image_labeled[x, :, :, :]))
        ax[0][0].set_ylabel('Ground Truth', fontsize=15)
        ax[0][0].set_xlabel('Sagital', fontsize=15)

        ax[0][1].imshow(np.rot90(image_labeled[:, y, :, :]))
        ax[0][1].set_xlabel('Coronal', fontsize=15)

        ax[0][2].imshow(np.squeeze(image_labeled[:, :, z, :]))
        ax[0][2].set_xlabel('Transversal', fontsize=15)

        ax[1][0].imshow(np.rot90(model_labeled_image[x, :, :, :]))
        ax[1][0].set_ylabel('Prediction', fontsize=15)

        ax[1][1].imshow(np.rot90(model_labeled_image[:, y, :, :]))
        ax[1][2].imshow(model_labeled_image[:, :, z, :])

        fig.subplots_adjust(wspace=0, hspace=.12)

        for i in range(2):
            for j in range(3):
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])

        return model_label_reformatted



    def z_normalize_img(self, img):
        """
        Normalize the image so that the mean value for each image
        is 0 and the standard deviation is 1.
        """
        for channel in range(img.shape[-1]):

            img_temp = img[..., channel]
            img_temp = (img_temp - np.mean(img_temp)) / np.std(img_temp)

            img[..., channel] = img_temp

        return img

    def crop(self, img, msk, randomize):
        """
        Randomly crop the image and mask
        """

        slices = []

        # Do we randomize?
        is_random = randomize and np.random.rand() > 0.5

        for idx in range(len(img.shape)-1):  # Go through each dimension

            cropLen = self.crop_dim[idx]
            imgLen = img.shape[idx]

            start = (imgLen-cropLen)//2

            ratio_crop = 0.20  # Crop up this this % of pixels for offset
            # Number of pixels to offset crop in this dimension
            offset = int(np.floor(start*ratio_crop))

            if offset > 0:
                if is_random:
                    start += np.random.choice(range(-offset, offset))
                    if ((start + cropLen) > imgLen):  # Don't fall off the image
                        start = (imgLen-cropLen)//2
            else:
                start = 0

            slices.append(slice(start, start+cropLen))

        return img[tuple(slices)], msk[tuple(slices)]

    def augment_data(self, img, msk):
        """
        Data augmentation
        Flip image and mask. Rotate image and mask.
        """

        # Determine if axes are equal and can be rotated
        # If the axes aren't equal then we can't rotate them.
        equal_dim_axis = []
        for idx in range(0, len(self.crop_dim)):
            for jdx in range(idx+1, len(self.crop_dim)):
                if self.crop_dim[idx] == self.crop_dim[jdx]:
                    equal_dim_axis.append([idx, jdx])  # Valid rotation axes
        dim_to_rotate = equal_dim_axis

        if np.random.rand() > 0.5:
            # Random 0,1 (axes to flip)
            ax = np.random.choice(np.arange(len(self.crop_dim)-1))
            img = np.flip(img, ax)
            msk = np.flip(msk, ax)

        elif (len(dim_to_rotate) > 0) and (np.random.rand() > 0.5):
            rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees

            # This will choose the axes to rotate
            # Axes must be equal in size
            random_axis = dim_to_rotate[np.random.choice(len(dim_to_rotate))]

            img = np.rot90(img, rot, axes=random_axis)  # Rotate axes 0 and 1
            msk = np.rot90(msk, rot, axes=random_axis)  # Rotate axes 0 and 1

        return img, msk


    def read_nifti_file(self, idx, randomize=False):
        """
        Read Nifti file
        """

        #idx = idx.numpy()
        #imgFile = self.filenames[idx][0]
        #mskFile = self.filenames[idx][1]
        #img = np.array(nib.load(imgFile).dataobj)
        #msk = np.array(nib.load(mskFile).dataobj)

        img, msk = self.load_example(idx.numpy())        

        if self.crop_dim[-1] == 1:
            img = np.rot90(img[..., [0]])  # Just take the FLAIR channel (0)
        else:
            img = np.rot90(img)

        msk = np.rot90(msk)

        """
        "labels": {
             "0": "background",
             "1": "edema",
             "2": "non-enhancing tumor",
             "3": "enhancing tumour"}
         """
        # Combine all masks but background
        if self.n_classes == 1:
            msk[msk > 0] = 1.0
            msk = np.expand_dims(msk, -1)
        else:
            msk_temp = np.zeros(list(msk.shape) + [self.n_classes])
            for channel in range(self.n_classes):
                msk_temp[msk == channel, channel] = 1.0
            msk = msk_temp

        # Crop
        img, msk = self.crop(img, msk, randomize)

        # Normalize
        img = self.z_normalize_img(img)

        # Randomly rotate
        if randomize:
            img, msk = self.augment_data(img, msk)

        return img, msk


    def get_train(self):
        """
        Return train dataset
        """
        return self.ds_train

    def get_test(self):
        """
        Return test dataset
        """
        return self.ds_test

    def get_valid(self):
        """
        Return validation dataset
        """
        return self.ds_val


    def get_dataset(self):
        """
        Create a TensorFlow data loader
        """

        ds = tf.data.Dataset.range(self.numFiles).shuffle(self.numFiles, 42)  # Shuffle the dataset

        ds_train = ds.take(self.len_train)
        ds_val_test = ds.skip(self.len_train)
        ds_val = ds_val_test.take(self.len_val)
        ds_test = ds_val_test.skip(self.len_val)

        ds_train = ds_train.map(lambda x: tf.py_function(self.read_nifti_file,
                                                         [x, True], [tf.float32, tf.float32]),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(lambda x: tf.py_function(self.read_nifti_file,
                                                     [x, False], [tf.float32, tf.float32]),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(lambda x: tf.py_function(self.read_nifti_file,
                                                       [x, False], [tf.float32, tf.float32]),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds_train = ds_train.repeat()
        ds_train = ds_train.batch(self.batch_size)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        batch_size_val = 4
        ds_val = ds_val.batch(batch_size_val)
        ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

        batch_size_test = 1
        ds_test = ds_test.batch(batch_size_test)
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

        return ds_train, ds_val, ds_test




class TFVolumeDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 list_IDs,
                 base_dir,
                 batch_size=1,
                 dim=(160, 160, 64),
                 n_channels=4,
                 n_classes=3,
                 shuffle=False,
                 augmentation=False,
                 verbose=0):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.base_dir = base_dir
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.verbose = verbose
        self.list_IDs = list_IDs
        self.augmentation = augmentation
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'

        # Initialization
        X = np.zeros((self.batch_size, *self.dim, self.n_channels),
                     dtype=np.float32)
        y = np.zeros((self.batch_size, *self.dim, self.n_classes),
                     dtype=np.float32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            if self.verbose == 1:
                print(f"Training on: {self.base_dir}{ID}")

            with h5py.File(self.base_dir + ID, 'r') as f:
                img = np.array(f.get("images")) #(4, 160, 160, 64)
                mask = np.array(f.get("masks")) #(3, 160, 160, 64)

                if self.n_channels == 1:
                    img = np.array(f.get("images"))
                    img = img[[0], ...]

                y[i] = np.rot90(np.moveaxis(mask, 0, 3))
                X[i] = np.rot90(np.moveaxis(img, 0, 3))
                
        return X, y



    def __data_augmentation(self, X, y):
        "Apply augmentation"
        X_augm, y_augm = augment_batch(X, y)
        return X_augm, y_augm


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        if self.augmentation:
            X, y = self.__data_augmentation(X, y)

        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)




def augment_data(img, msk):
    """
    Data augmentation
    Flip image and mask. Rotate image and mask.
    """
    
    # Determine if axes are equal and can be rotated
    # If the axes aren't equal then we can't rotate them.
    equal_dim_axis = []
    crop_dim = [160, 160, 64]
    for idx in range(0, len(crop_dim)):
        for jdx in range(idx+1, len(crop_dim)):
            if crop_dim[idx] == crop_dim[jdx]:
                equal_dim_axis.append([idx, jdx])  # Valid rotation axes
    
    dim_to_rotate = equal_dim_axis
    # make sure to use at least the 50% of original images
    if np.random.rand() > 0.5:
        # Random 0,1 (axes to flip)
        ax = np.random.choice(np.arange(len(crop_dim)-1))
        img = np.flip(img, ax)
        msk = np.flip(msk, ax)
    
    elif (len(dim_to_rotate) > 0) and (np.random.rand() > 0.5):
        rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees
        # This will choose the axes to rotate
        # Axes must be equal in size
        random_axis = dim_to_rotate[np.random.choice(len(dim_to_rotate))]
        img = np.rot90(img, rot, axes=random_axis)  # Rotate axes 0 and 1
        msk = np.rot90(msk, rot, axes=random_axis)  # Rotate axes 0 and 1

    return img, msk


def augment_batch(img_b, msk_b):
    batch_size = len(img_b)
    new_img_b, new_msk_b = np.empty_like(img_b), np.empty_like(msk_b)
    data_inputs = [(x, y) for x, y in zip(img_b, msk_b)]
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    res = pool.starmap(augment_data, data_inputs)
    for i in range(batch_size):
        new_img_b[i], new_msk_b[i] = res[i][0], res[i][1]
    pool.close()

    return new_img_b, new_msk_b


class VolumeDataset(torch.utils.data.Dataset):
  
    def __init__(self,
                 list_IDs,
                 base_dir,
                 batch_size=1,
                 shuffle=True,
                 dim=(160, 160, 64),
                 n_channels=4,
                 n_classes=4,
                 transform=False,
                 verbose=0
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.base_dir = base_dir
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes - 1
        self.verbose = verbose
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)


    def __getitem__(self, index):
        if self.verbose == 1:
            print(f"Training on: {self.base_dir}{index}")
        
        ID = self.list_IDs[index]

        with h5py.File(self.base_dir + ID, 'r') as f:
            X = np.array(f.get("images")) #(4, 160, 160, 64)
            y = np.array(f.get("masks")) #(3, 160, 160, 64)

        if self.transform:
            X, y = self.__data_augmentation(X, y)
            #data_aug = transforms.Compose([
            #    # spatial aug
            #    transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=0),
            #    transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=1),
            #    transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=2),
            #    # intensity aug
            ##    transforms.RandGaussianNoised(keys='image', prob=0.15, mean=0.0, std=0.33),
            ##    transforms.RandGaussianSmoothd(keys='image', prob=0.15, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
            ##    transforms.RandAdjustContrastd(keys='image', prob=0.15, gamma=(0.7, 1.3)),
            #])
            #X, y = data_aug({"image":X, "label":y})
        return X, y


def get_train_loader(args, dataset, set_type):
    if set_type == "train":
        data_ld = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                            num_workers=args.num_workers, pin_memory=True)
    else:
        data_ld = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                            num_workers=args.num_workers, pin_memory=True)
    return data_ld
