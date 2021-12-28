import os

from src.utils.model_utils import *
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop


class VAE:
    def __init__(self, dm, latent_dim=2, summary=False, learning_rate=0.0005, **kwargs):
        super(VAE, self).__init__(**kwargs)
        """

        :param img_shape: shape dell'immagine di ingresso al CVAE.
        :param latent_dim: dimensione del latent space nascosto.
        :param summary: Boolean, stampa summary
        :param learning_rate: learning rate per optimizer
        """
        self.dataManager = dm
        self.learning_rate = 0.0005
        self.img_shape = dm.img_shape
        self.latent_dim = latent_dim
        self.optimizer = None
        self.epoch = 0

        # ________________________________________________________________
        # INSTANZE DEI MODELLI DELL'AUTOENCODER:
        self.encoder, enc_input = Encoder(input_shape=self.img_shape, latent_dim=self.latent_dim).build_encoder()
        Z_ = self.encoder(enc_input)

        self.decoder, decoded = Decoder(input_shape=self.img_shape, latent_dim=self.latent_dim).build_decoder()
        decoder_output = self.decoder(Z_)

        self.model = Model(inputs=[enc_input], outputs=[decoder_output])
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()
        loss_fidelity = self.data_fidelity_loss(enc_input,decoder_output)
        self.model.add_loss(loss_fidelity)
        self.model.compile(optimizer=self._optimizer('adam'))

        # ________________________________________________________________
    def get_models(self):
        return self.model, self.encoder, self.decoder

    def train_autoencoder(self,
                          batch,
                          epochs,
                          data_training_x,
                          name_file_weights,
                          weights_path=str(),
                          re_train=False):
        # TODO sistemare il train dell'autoencoder, lavora sulle batch di immagini e non usare la funzione di fit ma
        #  train_on_batch
        files = os.listdir(weights_path)
        try:
            if name_file_weights in files and not re_train:
                file_weights = os.path.join(weights_path, name_file_weights)

                self.model.load_weights(file_weights)
                # self.save_model(path_run)
                print("INFO -----> PESI CARICATI DA: {}".format(weights_path))
                return
            else:
                print("INFO -----> Train Autoencoder")
                self.model.fit(data_training_x, batch_size=batch, epochs=epochs)
                self.model.save_weights(os.path.join(weights_path, name_file_weights))
        except Exception as e:
            print("| ERROR | --------> info:" + str(e))


    def train_(self, train_data,batch_size, epochs=1, ):
        """

        :param batch_size: dimensione della batch.
        :param epochs: numero di epoche per l'addestramento
        :param train_data: dataset per il training
        :param epochs, int: totale numero di epoche di addestramento.
        :return:
        """
        global dictionary
        # Addestramento e aggiornamento dei pesi della struttura, viene addestrata sulle epoche
        for epoch in range(self.epoch, self.epoch + epochs):
            time.time()
            start_time_epoch = time.time()
            print("\n_______ INIZIO ADDESTRAMENTO EPOCA:  %d" % (epoch))
            # quello che dobbiamo fare è aggiornare il la funzione obiettivo iterazione per iterazione
            # ________________________________________________________________
            # CHIAMATA A FUNZIONE DI TRAINING SU BATCH DI IMMAGINI_TRAINING (batch_size ,H ,W ,C)
            for step, (x_batch_train, _) in enumerate(train_data):
                x1, x2, x3 = self.train_step(x_batch_train)

            end_time_epoch = time.time()

            # ________________________________________________________________
            # Print informazioni di training
            print("||||| LOSS TOTALE: %d ;_______ DATA FIDELITY: %d ;_______ "
                  "kl_divergence_loss : %d" % (x1, x2,
                                               x3))  #  #
            # ________________________________________________________________  # 3) CREAZIONE AUTOENCODER


    def _optimizer(self, optimizer=str()):
        # CHIAMATA E INIZIALIZZAZIONE DI OPTIMIZER PER LA COMPILAZIONE DEL MODELLO

        optimizer = optimizer.lower()
        optimizers = ["adam", "rmsprop"]
        if optimizer not in optimizers:
            raise ValueError("Invalid mode type. Expected one of string value : %s" % optimizers)
        else:
            if optimizer == optimizers[0]:
                self.optimizer = Adam(learning_rate=self.learning_rate)
            elif optimizer == optimizers[1]:
                self.optimizer = RMSprop(learning_rate=self.learning_rate)
        return self.optimizer




    def data_fidelity_loss(self, X, X_hat, eps=1e-10):

        """
        :param eps: evita che il calcolo di log(1)
        :param Tensor X: dimensione delle immagini passate nel modello
        :param Tensor X_hat: dimensione delle immagini in uscita dal modello
        Metodo per inizializzare la reconstruction loss del modello. In pratica si tratta della prima parte della
        funzione obiettivo dell'autoencoder variazionale:
        componente LL della funzione ELBO, E_qϕ(z | x) [log p(x | z)] e ha una shape
        pari proprio alla (batch,)
        :return: Tensore con data_fidelity_loss, la funzione di perdita posta coma la cross Entropy loss tra la x
        predetta e la x
        originale

        E_{z ~ q_{phi}(z | x)}[log(p_{theta}(x|z))] = sum(x * log(x_hat) + (1 - x) * log(1 - x_hat)) binar
        """
        data_fidelity_loss = tf.losses.binary_crossentropy(K.flatten(X), K.flatten(X_hat)) * (self.img_shape[
                                                                                                  0]*self.img_shape[1])
        return K.mean(data_fidelity_loss)

    def loss(self, X, X_hat, log_std, mean):
        """

        :param Tensor X: Immagine originale passato nel VAE (N, H, W, C)
        :param Tensor X_hat: Immagine riconstruita (N ,H, W, C)
        :param log_std: output del layer standard deviation, uscita dell'encoder model (N, z_dim)
        :param mean: output del layer mean, uscita dell'encoder model (N, z_dim)

        :return: Un dizionario contentente tutte le funzioni di costo

        This method computes the loss of the VAE using the formula:
            L(x, x_hat) = - E_{z ~ q_{phi}(z | x)}[log(p_{theta}(x|z))]
                          + D_{KL}[q_{phi}(z | x) || p_{theta}(x)]

        """
        # richiamo le due loss function
        return self.data_fidelity_loss(X, X_hat) + self.kl_divergence_loss(X, X_hat)
