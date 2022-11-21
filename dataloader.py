import os
import json
import numpy as np
import nibabel as nib
import nilearn as nl
import tensorflow as tf
import matplotlib.pyplot as plt
import nilearn.plotting as nlplt
from tensorflow.keras.utils import to_categorical


class BratsDatasetGenerator:

    def __init__(self, args):

        self.ds_path = args.ds_path
        self.batch_size = args.batch_size
        self.crop_size = args.crop_size
        self.train_test_split = args.train_test_split
        self.val_test_split = args.val_test_split
        self.n_classes = args.n_classes
        self.seed = args.seed
        self.create_file_list()
        self.ds_train, self.ds_val, self.ds_test = self.get_dataset()


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

        self.filenames = {}
        for idx in range(self.numFiles):

            img_path = experiment_data["training"][idx]["image"][2:]
            label_path = experiment_data["training"][idx]["label"][2:]

            if img_path.split('/')[-1] == 'BRATS_001.nii.gz':
                print(idx)

            self.filenames[idx] = [os.path.join(self.ds_path,img_path),
                                    os.path.join(self.ds_path,label_path)]

            if idx == 0:
                self.img_shape = np.array(nib.load(self.filenames[idx][0]).get_fdata()).shape
                self.label_shape = np.array(nib.load(self.filenames[idx][1]).get_fdata()).shape


    def print_info(self):
        """ Print the dataset information """

        print("="*40)
        print(f"Dataset name:        {self.name}")
        print(f"Dataset description: {self.description}")
        print(f"Dataset release:     {self.release}")
        print(f"Dataset reference:   {self.reference}")
        print(f"Input channels:      {self.input_channels}")
        print(f"Output labels:       {self.output_channels}")
        print(f"Tensor image size:   {self.tensorImageSize}")
        print(f"Nbr of images:       {len(self.filenames)}")
        print(f"Image shape:         {self.img_shape}")
        print(f"Label shape:         {self.label_shape}")
        print("="*40)


    def load_one_example(self, idx):
        # load the image and label file, get the image content and return a numpy array for each
        image = np.array(nib.load(self.filenames[idx][0]).get_fdata())
        label = np.array(nib.load(self.filenames[idx][1]).get_fdata())
        return image, label

    def load_one_example_nl(self, idx):
        niimg = nl.image.load_img(self.filenames[idx][0])
        nimask = nl.image.load_img(self.filenames[idx][1])
        return niimg, nimask

    def plot_nl(self, idx):
        niimg, nimask = self.load_one_example_nl(idx)
        niimg = niimg.slicer[:, :, :, 0]
        fig, axes = plt.subplots(nrows=4, figsize=(8, 10))
        nlplt.plot_img(niimg, axes=axes[0])
        nlplt.plot_anat(niimg, axes=axes[1])
        nlplt.plot_epi(niimg, axes=axes[2])
        nlplt.plot_roi(nimask, bg_img=niimg,  axes=axes[3], cmap='Paired')
        plt.show()


    def standardize(image):
        """
        Standardize mean and standard deviation of each channel and z_dimension.

        Args:
            image (np.array): input image, shape (num_channels, dim_x, dim_y, dim_z)

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
                centered_scaled = centered / np.std(centered)
                standardized_image[c, :, :, z] = centered_scaled

        return standardized_image


    def get_sub_volume(self, idx, max_tries = 1000, background_threshold=0.965):
        """
        Extract random sub-volume from original images.

        Args:
            max_tries (int): maximum trials to do when sampling
            background_threshold (float): limit on the fraction
                of the sample which can be the background

        returns:
            X (np.array): sample of original image of dimension
                (num_channels, output_x, output_y, output_z)
            y (np.array): labels which correspond to X, of dimension
                (num_classes, output_x, output_y, output_z)
        """
        orig_x, orig_y = self.img_shape[0], self.img_shape[1]
        output_x, output_y = self.crop_size[0], self.crop_size[1]
        output_z = self.batch_size

        idx = idx.numpy()
        imgFile = self.filenames[idx][0]
        labFile = self.filenames[idx][1]
        image = np.array(nib.load(imgFile).get_fdata())
        label = np.array(nib.load(labFile).get_fdata())

        X = None
        y = None
        tries = 0

        while tries < max_tries:
            # randomly sample sub-volume by sampling the corner voxel (make sure to leave enough room for the output dimensions)

            start_x = np.random.randint(20 , (orig_x - output_x + 1)//2)
            start_y = np.random.randint(20 , (orig_y - output_y + 1)//2)
            start_z = np.random.randint(25 , 70)
            min_label_x = min(np.where(label != 0)[0])
            min_label_y = min(np.where(label != 0)[1])
            max_label_x = max(np.where(label != 0)[0])
            max_label_y = max(np.where(label != 0)[1])


            if (start_x < min_label_x) and (start_y < min_label_y) and (start_x + output_x > max_label_x) and (start_y+output_y > max_label_y):
                y = label[start_x: start_x + output_x,
                          start_y: start_y + output_y,
                          start_z: start_z + output_z]

                # One-hot encode the categories -> (output_x, output_y, output_z, num_classes)
                y = to_categorical(y, num_classes = self.n_classes)

                # compute the background ratio
                bgrd_ratio = np.sum(y[:,:,:,0]) / (output_x * output_y * output_z)

                tries += 1
                if bgrd_ratio < background_threshold:
                    X = np.copy(image[start_x: start_x + output_x,
                                      start_y: start_y + output_y,
                                      start_z: start_z + output_z, :])
                    # change dimension from (x_dim, y_dim, z_dim, num_channels)  to (num_channels, x_dim, y_dim, z_dim)
                    X = np.moveaxis(X,3,0)
                    # change dimension from (x_dim, y_dim, z_dim, num_classes) to (num_classes, x_dim, y_dim, z_dim)
                    y = np.moveaxis(y,3,0)
                    # take a subset of y that excludes the background class in the 'num_classes' dimension
                    y = y[1:, :, :, :]
                    X = self.standardize(X)
                    return X, y


    def get_training_set(self):
        ds = tf.data.Dataset.range(self.numFiles).shuffle(self.numFiles, self.seed) # Shuffle the dataset

        self.train_size = int(self.numFiles * self.train_val_split)
        ds_train = ds.take(self.train_size)
        ds_val_test = ds.skip(self.train_size)

        numValTest = self.numFiles - self.train_size
        self.val_size = int(numValTest * self.val_test_split)
        self.test_size = self.numFiles - (self.train_size - self.val_size)
        ds_val = ds_val_test.take(int(numValTest * self.val_test_split))
        ds_test = ds_val_test.skip(int(numValTest * self.val_test_split))


        ds_train = ds_train.map(lambda x: tf.py_function(self.get_sub_volume, [x], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(lambda x: tf.py_function(self.get_sub_volume, [x], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(lambda x: tf.py_function(self.get_sub_volume, [x], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds_train = ds_train.batch(self.batch_size)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        ds_val = ds_val.batch(self.batch_size)
        ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

        ds_test = ds_test.batch(1)
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

        return ds_train, ds_val, ds_test


    def get_training_sets(self):
        return self.ds_train, self.ds_val


    def get_test_set(self):
        return self.ds_test


    def visualize_patch(image, mask):
        # Flair channel
        image = image[0, :, :, :]
        # enhanced tumor
        mask = mask[2, :, :, :]
        print(image.shape)
        print(mask.shape)
        for i in range(image.shape[-1]):
            fig, ax = plt.subplots(1, 2, figsize=[10, 5], squeeze=False)
            ax[0][0].imshow(X[:, :, i], cmap='Greys_r')
            ax[0][0].set_yticks([])
            ax[0][0].set_xticks([])
            ax[0][0].set_title('MRI Flair')
            ax[0][1].imshow(mask[:, :, i], cmap='Greys_r')
            ax[0][1].set_xticks([])
            ax[0][1].set_yticks([])
            ax[0][1].set_title('Enhanced Tumor')
            fig.subplots_adjust(wspace=0, hspace=0)
            fig.suptitle(f'Inspection on depth {i}')
