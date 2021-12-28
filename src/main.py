import os

from data.data_utils.data_manager import Data_Manager_Mnist, Data_Manager_Claro_Slices
from src.models.vae import VAE
import tensorflow as tf


def main():
    # 1) path per il salvataggio

    # 2) Parametri di Train

    batch_size = 32
    epochs = 1000
    latent_dim = 2

    re_train = False  # set to true, se si vuole riaddestrare la rete. altrimente cambiare il nome dei pesi
    #l_mnist = Data_Manager_Mnist()
    l_slices = Data_Manager_Claro_Slices(64, 3)
    training = l_slices.X
    labels = l_slices.Y['ids']

    model = VAE(latent_dim=latent_dim, summary=True, dm=l_slices)

    train_dataset = tf.data.Dataset.from_tensor_slices(training)
    train_dataset = train_dataset.shuffle(buffer_size=128).batch(batch_size)



    model.train_autoencoder(batch_size, epochs, train_dataset, 'weights_1000.h5',
        'C:\\Users\\Ruffi\\Desktop\\Tesi_magistrale[XaI_GAN]\\model_saved\\weights')
    print("end")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


