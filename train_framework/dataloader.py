import os
import json
import cv2
import numpy as np
import nibabel as nib
import nilearn as nl
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import nilearn.plotting as nlplt
from utils import parse_args, set_seed
#from tensorflow.keras.utils import to_categorical

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
    #print(f"Train Images:     {np.shape(images)}  -- dtype: {images.dtype}")
    #print(f"Train masks:    {np.shape(masks)} -- dtype: {masks.dtype}")

    # Images are store as uint8 -> 0-255
    file.create_dataset("images", np.shape(images),h5py.h5t.STD_U8BE, data=images)
    file.create_dataset("masks", np.shape(masks),h5py.h5t.STD_U8BE, data=masks)
    file.close()
    return file


class BratsDatasetGenerator:

    def __init__(self, args):

        self.ds_path = args.ds_path
        self.batch_size = args.batch_size
        self.crop_size = args.crop_size
        self.train_val_split = args.train_val_split
        self.val_test_split = args.val_test_split
        self.n_classes = args.n_classes
        self.seed = args.seed
        self.create_file_list()
        #self.ds_train, self.ds_val, self.ds_test = self.get_dataset()


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
        self.numFiles = 16

        self.len_train = int(self.numFiles * self.train_val_split)
        self.val_test_len = self.numFiles - self.len_train
        self.len_val = int(self.val_test_len * self.val_test_split)
        self.len_test = self.numFiles - self.len_train - self.len_val
        self.train_ids = range(self.len_train)
        self.val_ids = range(self.len_train, self.len_train+self.len_val)
        self.test_ids = range(self.len_train+self.len_val, self.len_train+self.len_val+self.len_test)

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


    def load_example(self, idx):
        # load the image and label file, get the image content and return a numpy array for each
        image = np.array(nib.load(self.filenames[idx][0]).get_fdata())
        label = np.array(nib.load(self.filenames[idx][1]).get_fdata())
        return image, label


    def load_example_nl(self, idx):
        niimg = nl.image.load_img(self.filenames[idx][0])
        nimask = nl.image.load_img(self.filenames[idx][1])
        return niimg, nimask


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
                centered_scaled = centered / np.std(centered)
                standardized_image[c, :, :, z] = centered_scaled

        return standardized_image


    def generate_sub_volume(self, idx, max_tries = 150, background_threshold=0.96):
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
        orig_x, orig_y = self.img_shape[0], self.img_shape[1]
        output_x, output_y, output_z = self.crop_size[0], self.crop_size[1], self.crop_size[2]

        #idx = idx.numpy()
        image, label = self.load_example(idx)
        X = None
        y = None
        tries = 0

        while tries < max_tries:
            # randomly sample sub-volume by sampling the corner voxel
            # (make sure to leave enough room for the output dimensions)
            min_x = min(np.where(label != 0)[0])
            max_x = max(np.where(label != 0)[0])
            min_y = min(np.where(label != 0)[1])
            max_y = max(np.where(label != 0)[1])
            min_z = min(np.where(label != 0)[2])
            max_z = max(np.where(label != 0)[2])
            len_z = max_z - min_z

            start_x = (orig_x - output_x + 1)//2
            start_y = (orig_y - output_y + 1)//2

            if start_x > min_x - 8:
                start_x = min_x - 8
            if start_y > min_y - 8:
                start_y = min_y - 8

            if (max_z - min_z <= 60):
                if (max_z - min_z <= 32):
                    start_z = min_z
                else:
                    start_z = np.random.randint(min_z, max_z-output_z)

                #print(f"Start X: {start_x}, end_X: {start_x+output_x}, min_x: {min_x}, max_x: {max_x}")
                #print(f"Start Y: {start_y}, end_Y: {start_y+output_y}, min_y: {min_y}, max_y: {max_y}")
                #print(f"Start Z: {start_z}, end_Z: {start_z+output_z}, min_z: {min_z}, max_z: {max_z}, len_Z: {max_z-min_z}")
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
                # take a subset of y that excludes the background class in the 'n_classes' dimension
                y = y[1:, :, :, :]
                X = self.standardize(X)
                print(y.shape)
                return X, y, start_x, start_y, start_z

            else:
                start_z = np.random.randint(min_z, max_z-output_z)

            if len_z <= 75:
                background_threshold=0.9865
            elif len_z > 75 and len_z <= 85:
                background_threshold=0.969

            if (start_x + output_x > max_x) and (start_y+output_y > max_y):

                y = label[start_x: start_x + output_x,
                          start_y: start_y + output_y,
                          start_z: start_z + output_z]

                # One-hot encode the categories -> (output_x, output_y, output_z, n_classes)
                y = np.eye(self.n_classes, dtype='uint8')[y.astype(int)]

                # compute the background ratio
                bgrd_ratio = np.sum(y[:,:,:,0]) / (output_x * output_y * output_z)

                tries += 1
                print(f"background_ratio: {bgrd_ratio}")
                if bgrd_ratio < background_threshold:
                    #print(f"Start X: {start_x}, end_X: {start_x+output_x}, min_x: {min_x}, max_x: {max_x}")
                    #print(f"Start Y: {start_y}, end_Y: {start_y+output_y}, min_y: {min_y}, max_y: {max_y}")
                    #print(f"Start Z: {start_z}, end_Z: {start_z+output_z}, min_z: {min_z}, max_z: {max_z}, len_Z: {max_z-min_z}")
                    X = np.copy(image[start_x: start_x + output_x,
                                      start_y: start_y + output_y,
                                      start_z: start_z + output_z, :])
                    # (x_dim, y_dim, z_dim, n_channels) -> (n_channels, x_dim, y_dim, z_dim)
                    X = np.moveaxis(X,3,0)
                    # (x_dim, y_dim, z_dim, n_classes) -> (n_classes, x_dim, y_dim, z_dim)
                    y = np.moveaxis(y,3,0)
                    # take a subset of y that excludes the background class in the 'n_classes' dimension
                    y = y[1:, :, :, :]
                    X = self.standardize(X)
                    print(y.shape)
                    print("\n -------------------- \n")
                    return X, y, start_x, start_y, start_z

        print("No valid sub volume")
        print(f"Start Z: {start_z}, end_Z: {start_z+output_z}, min_z: {min_z}, max_z: {max_z}, len_Z: {max_z-min_z}")


    def create_volume_sets(self):
        train_dir_name = '../resources/BRATS_ds/Train'
        val_dir_name = '../resources/BRATS_ds/Validation'
        test_dir_name = '../resources/BRATS_ds/Test'

        dir_paths = [train_dir_name, val_dir_name, test_dir_name]
        id_set_lst = [self.train_ids, self.val_ids, self.test_ids]
        set_lens = [self.len_train, self.len_val, self.len_test]

        ds_dict = dict()
        for dir_name, id_lst, s_len in zip(dir_paths, id_set_lst, set_lens):
            files_dict = dict()
            set_type = dir_name.split('/')[-1]

            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            for i in id_lst:
                name = f"BRATS_{i}"
                #image, label, start_x, start_y, start_z = self.generate_sub_volume(tf.constant(i))
                image, label, start_x, start_y, start_z = self.generate_sub_volume(i)
                name = name + f"_{start_x}_{start_y}_{start_z}.h5"
                path_name = os.path.join(dir_name, name)
                files_dict[i] = name
                store_hdf5(path_name, image, label)
            print("\n -- SET DONE ---")
            ds_dict[set_type] = {"len": s_len, "files":files_dict}

        with open('../resources/BRATS_ds/config.json', 'w') as f:
            json.dump(ds_dict, f, indent=4)


    def get_sub_volume(self, idx, config, set_type):
        files = config[set_type]['files']
        cur_file = files[str(idx)][:-3]
        start_x = int(cur_file.split('_')[-3])
        start_y = int(cur_file.split('_')[-2])
        start_z = int(cur_file.split('_')[-1])

        img, label = self.load_example(idx)

        img = img[start_x: start_x+self.crop_size[0],
                start_y: start_y+self.crop_size[1],
                start_z: start_z+self.crop_size[2], :]

        mask = np.eye(self.n_classes, dtype='uint8')[label.astype(int)]
        #to_categorical(label, num_classes = self.n_classes)

        mask = mask[start_x: start_x + self.crop_size[0],
                    start_y: start_y + self.crop_size[1],
                    start_z: start_z + self.crop_size[2]]

        # change dimension from (x_dim, y_dim, z_dim, n_channels)  to (n_channels, x_dim, y_dim, z_dim)
        img = np.moveaxis(img,3,0)
        # change dimension from (x_dim, y_dim, z_dim, n_classes) to (n_classes, x_dim, y_dim, z_dim)
        mask = np.moveaxis(mask,3,0)
        # take a subset of y that excludes the background class in the 'n_classes' dimension
        mask = mask[1:, :, :, :]
        img = self.standardize(img)
        return img, mask


    def plot_nl(self, idx):
        niimg, nimask = self.load_example_nl(idx)
        niimg = niimg.slicer[:, :, :, 0]
        fig, axes = plt.subplots(nrows=4, figsize=(8, 10))
        nlplt.plot_img(niimg, axes=axes[0])
        nlplt.plot_anat(niimg, axes=axes[1])
        nlplt.plot_epi(niimg, axes=axes[2])
        nlplt.plot_roi(nimask, bg_img=niimg,  axes=axes[3], cmap='Paired')
        plt.show()


    def plot_example(self, idx, d_chan):
        img, label = self.load_example(idx)
        img = img[:, :, :, 0]
        plt.figure("image", (18, 6))
        plt.subplot(1, 2, 1)
        plt.title("image")
        plt.imshow(img[:, :, d_chan], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title("label")
        plt.imshow(label[:, :, d_chan])
        plt.show()


    def colorize_labels_image_flair(self, idx):
        image, label = self.load_example(idx)
        #label = to_categorical(label, num_classes=4).astype(np.uint8)
        label = np.eye(self.n_classes, dtype="uint8")[label.astype(int)]
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


    def plot_image_grid(self, idx):
        image = self.colorize_labels_image_flair(idx)

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


    def visualize_patch(self, image, mask, m_chan=0):
        # Flair channel
        image = image[0, :, :, :]
        # Edema tumor
        mask = mask[m_chan, :, :, :]
        for i in range(image.shape[-1]):
            plt.figure("image", (8, 6))
            plt.subplot(1, 2, 1)
            plt.title('MRI Flair Channel')
            plt.imshow(image[:, :, i], cmap="gray")
            plt.subplot(1, 2, 2)
            plt.title('Edema Channel')
            plt.imshow(mask[:, :, i])
            plt.suptitle(f'Inspection on depth {i}')
            plt.show()


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

    #def get_dataset(self):
    #    ds = tf.data.Dataset.range(self.numFiles).shuffle(self.numFiles, self.seed) # Shuffle the dataset
    #    ds_train = ds.take(self.len_train)
    #    ds_val_test = ds.skip(self.len_train)
    #    ds_val = ds_val_test.take(int(self.val_test_len * self.val_test_split))
    #    ds_test = ds_val_test.skip(int(self.val_test_len * self.val_test_split))
    #    ds_train = ds_train.map(lambda x: tf.py_function(self.generate_sub_volume, [x], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #    ds_val = ds_val.map(lambda x: tf.py_function(self.generate_sub_volume, [x], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #    ds_test = ds_test.map(lambda x: tf.py_function(self.generate_sub_volume, [x], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #    ds_train = ds_train.batch(1)
    #    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    #    ds_val = ds_val.batch(1)
    #    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)
    #    ds_test = ds_test.batch(1)
    #    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    #    return ds_train, ds_val, ds_test
    #def get_training_sets(self):
    #    return self.ds_train, self.ds_val
    #def get_test_set(self):
    #    return self.ds_test




class TFVolumeDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 list_IDs,
                 base_dir,
                 batch_size=1,
                 shuffle=True,
                 dim=(160, 160, 32),
                 n_channels=4,
                 n_classes=4,
                 verbose=1):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.base_dir = base_dir
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes - 1
        self.verbose = verbose
        self.list_IDs = list_IDs
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'

        # Initialization
        X = np.zeros((self.batch_size, self.n_channels, *self.dim),
                     dtype=np.float64)
        y = np.zeros((self.batch_size, self.n_classes, *self.dim),
                     dtype=np.float64)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if self.verbose == 1:
                print("Training on: %s" % self.base_dir + ID)
            with h5py.File(self.base_dir + ID, 'r') as f:
                X[i] = np.array(f.get("images"))
                # remove the background class
                y[i] = np.array(f.get("masks"))
                print(y[i].shape)
                #y[i] = np.moveaxis(y[i], 3, 0)[1:]
                #y[i] = np.moveaxis(np.array(f.get("masks")), 3, 0)[1:]

        return X, y


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[
                  index * self.batch_size: (index + 1) * self.batch_size]

        print(f"indexes get item: {indexes}")
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        print(f"indexes on ep end: {self.indexes}")
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            print(f"indexes on ep end shuffle: {self.indexes}")


## ADD PYTORCH DATADLOADER
#class PTVolumeDataGenerator(torch.utils.data.Dataset):
#  'Characterizes a dataset for PyTorch'
#  def __init__(self, list_IDs, labels):
#        'Initialization'
#        self.labels = labels
#        self.list_IDs = list_IDs
#
#  def __len__(self):
#        'Denotes the total number of samples'
#        return len(self.list_IDs)
#
#  def __getitem__(self, index):
#        'Generates one sample of data'
#        # Select sample
#        ID = self.list_IDs[index]
#
#        # Load data and get label
#        X = torch.load('data/' + ID + '.pt')
#        y = self.labels[ID]
#
#        return X, y


def main():

    args = parse_args()
    set_seed(args)

    brats_generator = BratsDatasetGenerator(args)
    brats_generator.print_info()
    args.class_names = [v for k, v in brats_generator.output_channels().items()]
    print(f"  Class names = {args.class_names}")

    brats_generator.create_volume_sets()


if __name__ == "__main__":
    main()
