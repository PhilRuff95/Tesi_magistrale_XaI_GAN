from __future__ import absolute_import
from math import log2

from keras.models import Model
from tensorflow.keras.layers import Layer, LeakyReLU, Conv2DTranspose, Conv2D, Flatten, Input, Reshape, Dense, ReLU, \
    Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


class Reparametrize(Layer):

    def __init__(self, **kwargs):
        super(Reparametrize, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Reparametrize, self).build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Trick di riparametrizzazione per poter garantire l'aggiornamento dei pesi della rete fino al layer di input
        dell'encoder.
        Inputs : mean = Z_mu , std = esp ( .5  Z_logvar )
        Outputs : z_sample = tramite un trick di riparametrizzazione, tramite una distribuzione normale
                    epsilon, N (0, I) : si ricostruisce lo spazio z = mu + eps * std


        """
        # dai layer densi della media e della varianza : inputs

        Z_mu, Z_logvar = inputs

        # dimensione della batch
        batch = K.shape(Z_mu)[0]

        # dimensione del vettore si a Z_mean sia di Z_logvar
        dimension = K.int_shape(Z_mu)[1]

        # ________________________________________________________________
        # Add Divergency Loss to model.
        loss_kl = self.kl_divergence_loss(Z_mu, Z_logvar)
        self.add_loss(losses=[loss_kl])

        epsilon = K.random_normal(shape=(batch, dimension))

        sigma = K.exp(.5 * Z_logvar)

        return Z_mu + sigma * epsilon

    def kl_divergence_loss(self, mean, log_std):
        """
        :param Tensor mean: tensore della media in uscita dal layer Z_mu dell'encoder, dimensione (N, z_dim)
        :param Tensor std:  tensore della deviazione standard in uscita dal layer Z_std dell'encoder, dim. (N,z_dim)
        :return: tensore dato dalla formula della stima di Montecarlo della divergenza kl

         D_{KL}[q_{phi}(z | x) || p_{theta}(x)] = (1/2) * sum(-std + -mean^2 + 1 - log(std))

         Qui si tratterebbe della stima di Kl solo per 1 campione della batch, quindi in pratica, viene calcolata la
         std e mean in uscita dall'encoder, questi vengono poi inseriti nella equazione di cui sopra:
         In uscita si otterra una quantità sommata su tutta la batch.
         Nell'equazione viene sostituito il valore di std con exp^(std) per una maggiore stabilità.

        """
        kl_divergence_loss = (-1 / 2) * K.sum((1 + log_std - mean ** 2 - K.exp(log_std)), axis=[1])

        return K.mean(kl_divergence_loss)


get_custom_objects().update({'SampleLayer': Reparametrize})


# ________________________________________________________________
# CREAZIONE DEI MODELLI PER AUTOENCODER VARIAZIONALE
# ________________________________________________________________
# 1) CREAZIONE ENCODER
class Functional_model(object):
    def __init__(self, input_shape, latent_dim=2, leaky=False):
        self._input_shape = input_shape
        self._latent_dim = latent_dim
        self._leaky = leaky
        self._reps = int(log2(input_shape[1] / 4))

    def _conv_deconv(self, input_layer, name):
        raise NotImplementedError("| Error | --------------> Non imolementata nella classe parente")

    def build(self):
        raise NotImplementedError("| Error | --------------> Non imolementata nella classe parente")


class Encoder(Functional_model):

    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        """
                ------------------------
                Build Encoder Probabilistico:
                L'encoder qui è una struttura CNN con layer down-sampling, la sostanza è che si
                tratta di eleaborare una
                stima della pobabilità a posteriori p(z | x) , che nel caso della verosmiglianza logaritmica
                marginale nei
                DLVM si tratta di un problema non computazionalmente trattabile. Per questo motivo si approssima la si
                approssima con un altra distribuzione q (z | x) , appunto l'inference model/ Encoder. Si usa la
                divergenza KL
                applicata alle due distribuzioni e si ottiene il problema della massimizzazione ella funzione obiettivo:
                EVIDENCE-LOWER-BOUND(ELBO)
                qϕ viene scelta come un infinite mixture of multivariate Gaussians distributions, per ogni punto si
                ottiene
                un certo vettore mu e una matrice di covarianza. Quindi il latent code code è un mix di Gaussiane.

                :param latent_dim: dimensione del collo di bottiglia.
                :param input_shape: DIMENSIONE DI INGRESSO ALLA RETE DELLE IMMAGINI ( N * N * channels)
                :return:keras.Model, GLI OUTPUT SI OTTENGONO ATTRAVERSO DUE DENSI, qϕ(z | x) = N(μ(x),diag(σ^2(x))), la
                matrice di covarianza diagonale, e il vettore delle medie.

                """

    def _conv_deconv(self, input_layer, name):
        x = Conv2D(filters=32, kernel_size=4, strides=2, padding='same', name=name)(input_layer)
        x = get_activation('leaky')(x) if self._leaky else get_activation('relu')(x)
        return x

    def build(self):
        init = x = Input(shape=self._input_shape)
        for i in range(self._reps):
            name = 'conv_layer_{}'.format(i)
            x = self._conv_deconv(x, name)

        x = Flatten()(x)

        x = Dense(units=512)(x)
        x = Activation('leaky')(x) if self._leaky else get_activation('relu')(x)

        x = Dense(units=256)(x)
        x = Activation('leaky')(x) if self._leaky else get_activation('relu')(x)

        # Separazione in due densi, dal layer denso in Z_mu e Z_logvar con dimensioni pari al latent_dim.
        Z_mu = Dense(self._latent_dim, name="Z_mu_dense")(x)
        Z_logvar = Dense(self._latent_dim, name="Z_logvar_dense")(x)
        Z_ = Reparametrize(name="Reparametrize_layer")([Z_mu, Z_logvar])

        return init, Z_

    def build_encoder(self) -> object:
        init, Z_ = self.build()
        # Inizializzazione del modello Encoder
        return Model(init, Z_, name="Encoder_Model"), init


# 2) CREAZIONE DECODER
class Decoder(Functional_model):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self._decoder_output = self._input_shape
        """
        ------------------------
        Build Decoder probabilistico:
        Struttura upsampling CNN, si tratta della componente generativa dell'autoencoder. Il decoder prende in
        ingresso la viariabile z (z1,_,zn) in uscita dal encoder. La struttura dello spazio latente si basa su un
        prior, p(z) ovvero una distribuzione a priori che il VAE impone allo spazio latente, qui si tratta di una
        distribuzione p (z) = N ( 0, I ), con varianze unitarie e in forma diagonale . p(x | z) apprende a
        la joint probability p(x,z) = p(z) p(x|z), con una distribuzione a priori del latent code z e un decoder
        stocastico p (x|z).
        :return: Spazio di partenza del Dataset ,immagine ricostruita x_tilde
        """

    def _conv_deconv(self, input_layer, pad_mode="same"):
        x = Conv2DTranspose(32, 4, strides=2, padding=pad_mode)(input_layer)
        x = get_activation('leaky')(x) if self._leaky else get_activation('relu')(x)
        return x

    def build(self):

        decoder_input = Input(shape=(self._latent_dim,), name="decoder input")
        x = Dense(units=256)(decoder_input)
        x = get_activation('leaky')(x) if self._leaky else get_activation('relu')(x)
        x = Dense(units=512)(x)
        x = get_activation('leaky')(x) if self._leaky else get_activation('relu')(x)

        x = Reshape(target_shape=(-1, 1, 512), input_shape=(512,))(x)
        for i in range(self._reps):
            if i == 0:

                x = self._conv_deconv(x, 'valid')
            else:

                x = self._conv_deconv(x)
        x = Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(x)
        decoded = x
        return decoded, decoder_input

    # senza questo layer di attivazione l'uscita diventa un logit ( una probability ): si gestisce meglio in fase
    # di addestramento della data_fidelity_loss, utilizzando una binary_crossentropy_with_logits()
    # _______________________________________________________________
    # Inizializzazione del modello Decoder
    def build_decoder(self):
        decoded, decoder_input = self.build()
        return Model(decoder_input, decoded,
            name="Decoder"), decoded  # ________________________________________________________________


# ________________________________________________________________

def get_activation(activation):
    """
    activation = ['relu', 'sigmoid', 'leaky']


    :param activation: specifica che tipo di Lahyer di attivazione si vuole utilizzare, ci sono de possibilità:
                        "relu" : significa che l'attivazione è di tipo a rettificatore,
                        |1) se il valore di ingresso è < del thresholf l'uscita è zero
                        |2) se il val. è compreso tra th e x-max , l'uscita è x.
                        |3) se il valore è > del x_max, l'output è x_max
                        "sigmoid": ativazione a sigmoide, applica la funzione:
                        |σ(x) = 1/(1+exp(-x)) ,f(x) = [0,1]
                        |D = (-inf, inf)
                        | f(0) = 0.5
                        "leaky": leaky relu,  a pendenza nell'asse negativo.

    :return: Layer di attivazione
    """

    if activation == "relu":
        return ReLU()
    elif activation == "sigmoid":
        return Activation(activation)
    elif activation == "leaky":
        return LeakyReLU()
