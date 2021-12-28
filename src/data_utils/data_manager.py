from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import urllib.request
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy import io


class DataManager(object):
    def __init__(self, normalize_0_1=False, normalize_1_1=False):
        # ________________________________
        # BOOLEANS:
        self.normalize_0_1 = normalize_0_1
        self.normalize_1_1 = normalize_1_1
        self.rgb = False
        self.inizialized = False
        # ________________________________
        # STRINGS: folders ed altre stringhe importanti per il Data_manager
        self.name = str()
        self.file_name = str()
        self.data_folder = 'C:/Users/Ruffi/Desktop/Tesi_magistrale[XaI_GAN]/data/data_'

        # ________________________________
        # ARRAYS, LIST, DICT
        self.X = None
        self.Y = None
        self.labels = list
        self.info_dataset = dict()
        self.img_shape = list()

    def set_inizialized(self):

        if self.inizialized:
            return
        else:
            self.inizialized = True

    def check_rgb(self):
        if len(self.X.shape[1:]) < 3:
            self.rgb = False
        else:
            self.rgb = True

    def Load_data(self):

        """
        funzione di caricamento del dataset: Implementata nella classe madre, sovrascritta nelle altre classi figlie.
        :return: ritorna i dati caricati dai folder_path di riferimento

        1) Mnist dataset digits
        2) Crops slices, 121 pazienti, 90% nan over.
        """
        try:
            self._load_data() if not self.inizialized else print("| INFO | --------> {} precedentemente "
                                                                 "caricato nella classe, data_Manager".format(
                self.name))
            pass
        except Exception as e:
            print("| ERROR | --------> info:" + str(e))

    def compute_labels(self, array):
        """
        Calcolo delle labels unique e il numero di volte che vengono richiamate
        :return:
        """
        values, counts = np.unique(array, return_counts=True)
        number_of_labels = len(values)
        return tuple([values, counts, number_of_labels])

    def summary(self):
        border = 5
        intro_str = "|" * border + "_____SUMMARY DEL:  {} _____".format(self.name) + '|' * border
        print(intro_str)
        for key, item in self.info_dataset.items():
            print('|' * border + '_' * (len(intro_str) - 2 * border) + '|' * border)
            str_middle = '|' * border + key.replace("_", " ") + ' =: ' + str(item)
            print(str_middle + (len(intro_str) - len(str_middle) - border) * ' ' + '|' * border)
        print('|' * len(intro_str))


class Data_Manager_Mnist(DataManager):
    def __init__(self, batch_size=32, train_size=70000, **kwargs):
        """

        :param batch_size:
        :param color:  Boolean type, se i channel del colori sono 1 (B, W), o a tre canali (R, G, B)
        """
        super(Data_Manager_Mnist, self).__init__(**kwargs)
        self.batch_size = batch_size  # Batch size
        # self.color = color  in questo caso il colore non c'è
        self.data_shape = None
        self.train_size = train_size
        self.name = " Mnist digits Dataset"
        self.file_path = self.data_folder + '/mnist/mnist_/archive/mnist.npz'
        # ________________________________________________________________
        # Load data dopo aver inizilizzato tutti gli attributi:
        self.Load_data()
        print("| INFO | --------> {} caricato nella classe : data_Manager".format(self.name))

    def _load_data(self):
        """
         ________________________________________________________________
         Carico la cartella compressa di digits mnist. il file viene caricato e poi devono essere richiamate ogni
         singole subfold compresse: shape-> (len. , H , W )
         dtype = ndarray
         ________________________________________________________________
         - x_train : shape= ( 60000, 28, 28 )
         - x_test : shape= ( 10000, 28, 28 )
         - labels : shape= ( 10 )
         - y_train : shape = (60000, 1 )
         - y_test : shape = (10000, 1 )
        :return: carica il dataset nel sistema
        """

        f_loaded = np.load(self.file_path)

        x_train, y_train = f_loaded['x_train'], f_loaded['y_train']
        x_test, y_test = f_loaded['x_test'], f_loaded['y_test']
        # ________________________________________________________________
        self.X = {"train": self.preprocess_images_0_1(x_train) if self.normalize_0_1 else x_train,
                  "test": self.preprocess_images_0_1(x_test) if self.normalize_0_1 else x_test}
        self.Y = {"train": y_train, "test": y_test}

        # ________________________________________________________________
        # Computiamo le labels del training set
        self.labels = self.compute_labels(y_train)

        # ----------------------------------------------------------------  #
        # Calcolo delle informazioni generali del  dataset Mnist
        self.info_dataset = self.get_dataset_info()

        self.summary()
        self.set_inizialized()

    def get_concatenated_x(self):
        """

        :return: sia il training set e il test set concatenati insieme sullo stesso asse, quindi la dimensione del
        dato muta. da


        (Ntr, H, W, ch) ; (Nte, H, W, ch) ----> (Ntr + Nte, H, W, ch)
        """
        x_tr, y_tr = self.get_training_set()
        x_te, y_te = self.get_test_set()
        return np.concatenate((x_tr, x_te), axis=0), np.concatenate((x_tr, x_te), axis=0)

    def set_X(self, key, array):
        self.X[key] = array

    def get_training_set(self):
        return self.X["train"], self.Y["test"]

    def get_test_set(self):
        return self.X["test"], self.Y["test"]

    def get_dataset_info(self):
        """
        return all the information about the dataset
        :return: dictionary
        """

        train_shape = self.X["train"].shape
        test_shape = self.X["test"].shape
        _shape = train_shape[1:]

        assert train_shape[1:] == test_shape[1:]
        # scorriamo tutto l'array e inseriamo una nuova dimensione nel
        if len(_shape) < 3:
            # se la shape della train non ha più di tre dimensioni significa che non ha una dimensione associata ai
            # canali RGB, quindi per questo dobbiamo aggiungere la dimensione del singolo canale (W/B)
            for key, x in self.X.items():
                print(' | INFO | ------> reshaping delle immagini di  : {}  , immagini ad 1 canale'.format(key))
                x = np.asarray([img[:, :, np.newaxis] for img in x])
                self.set_X(key, x)
        else:

            print(' | INFO | ------> reshaping delle immagini non necessario, numero di canali  : {}  '.format(
                _shape[-1]))
        train_shape = self.X["train"].shape
        test_shape = self.X["test"].shape

        self.img_shape = [el for el in train_shape[1:]]
        _size = (train_shape[0], test_shape[0])  # Numero di campioni per set di train, test ( Ntr, Nte ) , tuple
        return {"total_size": sum(_size), "train_size": _size[0], "test_size": _size[1], "labels ": self.labels[2],
                "img_shape": train_shape[1:]}

    # ________________________________________________________________
    # Preprocessing del Dataset

    def preprocess_images_0_1(self, images):
        """
        Preprocessing per i digits 32*32, scala le immagini tra (0,1)
        - A questo livello si cercano i massimi pixel per pixel.
        :param images: immagini da processare, in genere si prendono tutto un arrray in ingresso
        :param max:  il massimo valore sia nell'asse 0 che nell'asse 1, (x , y) pixel*pixel
        :return: ritorna le immagini processate in formato float32
        """
        # Cerco i valori massimi rispetto ai due assi : x e y , pixel*pixel

        if len(images.shape) > 2:
            max_x = np.amax(np.amax(images, axis=0))
            max_y = np.amax(np.amax(images, axis=1))
            max_ = max_x if max_x >= max_y else max_y
            return np.expand_dims(images, axis=-1).astype(
                'float32') / max_  # scala le immagini tra 0 e 1 ( normalizazzione ) e le riporta in float32
        else:
            raise ValueError("--- Errore inserimento line 177 preprocess_images")


class Data_Manager_Claro_Slices(DataManager):
    def __init__(self, image_H_W_to_load, channels_to_load, **kwargs):

        super(Data_Manager_Claro_Slices, self).__init__()

        # _______________________________________________________________
        # Informazioni necessarie per caricare i file
        self.image_H_W_to_load = image_H_W_to_load
        self.channels = channels_to_load if channels_to_load == 3 else 1
        self.file_path = self.data_folder + '/claro/data_raw/claro/{}/{}/' \
                                            '_nan_under_90'.format(self.channels, self.image_H_W_to_load)
        self.sliding_windows_file = self.file_path + '/sliding_windows_clean.mat'
        self.residual_ID_paziente_file = self.file_path + '/residual_id_paziente.mat'
        self.name = " Dataset Claro_Slices---> (Dimensione immagini:  {} x {} , {} ),".format(self.image_H_W_to_load,
            self.image_H_W_to_load, self.channels)

        # ________________________________________________________________
        # Loading slices
        self.Load_data()

    def _load_data(self):
        """
        ________________________________________________________________
        Funzione per caricare le slices 2D dal file.mat. Ci sono due file che a noi interessano in particolare:
        1) Il file in cui sono contenute le immagini
               Returns:
                       A dictionary with three fields:
                       'img': [(N_slices, height, width, ch)] numpy array of dtype float32 containing the 2D slices of
                       all
                       patients
                       'id': [(N_slices, 1)] numpy array of Unicode string of maximum length 10 (U10) representing
                       patients id associated with each slices
                       'label': [(N_slices, 1)] numpy array of dtype uint8 (in the case of classification)
                        representing the patients label associated with each slices
               """
        _load_slices = io.loadmat(self.sliding_windows_file)
        _load_residual_id = io.loadmat(self.residual_ID_paziente_file)

        # ________________________________________________________________
        # Residual ID (121, 1)
        #  - id_paziente = array (1,), ( dtype = <U5-U8 )
        #  - label_class = array (1,), ( dtype = uint8)
        #  - numero di immagini per id= array (1,), (dtype = uint8 )

        residual_ID_ = _load_residual_id['residual_ID']
        _data = _load_slices['sliding_windows_clean']
        # Carico tutto il dataset con 3 campi : ( slices, id, label ADAPTIVE 1/0 )
        # ________________________________________________________________
        # 1) carico le slices. al cui interno ci sono i tre campi che ci interessano.
        # successiva rielaborazione.
        slices = _data[0]
        slices_data = np.asarray([self.loader(slices[idx][0]) for idx in range(slices.shape[0])], dtype='float32')
        self.X = slices_data
        self.img_shape= [el for el in self.X.shape[1:]]
        # come risultato finale si ottiene un : ndarray: (N_img, H, W, Ch )
        # 2) label Adaptive/Non_Adaptive (0, 1)
        slices_label = np.asarray([slices[idx][2][0] for idx in range(slices.shape[0])], dtype='uint8')


        # 3) Id_paziente (U5-U8 )
        slices_id = np.asarray([slices[idx][1] for idx in range(slices.shape[0])], dtype='<U10')
        self.Y = {'label_class': slices_label, 'ids': slices_id}
        print("| INFO | --------> {} caricato nella classe, data_Manager".format(self.name))

    def loader(self, img, fill_nan=-1000):
        """

        :param img: immagine del dataset che deve essere rimaneggiata.
        :param fill_nan: valore che deve essere sostituito al posto dei nan
        :return: l'immagine stessa
        """

        img = np.array(img)

        if np.isnan(img).any():  # remove nan
            mask = np.isnan(img)
            img[mask] = fill_nan
        assert ~np.isnan(img).any()

        if img.ndim < 3:
            img = img[:, :, np.newaxis]

        if self.normalize_0_1:  # rescale data tra 0/1
            img = self.rescale_m0_p1(img)
        if self.normalize_1_1:  # rescale data tra -1/1
            img = self.rescale_m1_p1(img)

        img = tf.cast(img, tf.float32)  # cast del file img come float 32
        return img

    def rescale_m1_p1(self, img):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = (img - 0.5) * 2
        return img

    def rescale_m0_p1(self, img):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        return img

    def data_iterators(self):
        #TODO scrivere la funzione per inserire un data iterator nella classe.
        print("")


