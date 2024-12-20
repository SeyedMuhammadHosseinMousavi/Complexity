import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.losses import mse
import tensorflow.keras.backend as K
import time
from memory_profiler import memory_usage

# Variational Autoencoder for Synthetic Data Generation
def sampling(args):
    """Reparameterization trick for sampling latent variables"""
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# VAE Pipeline
def vae_synthetic_data():
    # Load Iris dataset
    iris = load_iris()
    data = iris.data

    # Scale data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # Hyperparameters
    latent_dim = 2  # Dimensionality of latent space
    input_dim = data.shape[1]
    epochs = 100
    batch_size = 32

    # Define encoder
    inputs = Input(shape=(input_dim,))
    h = Dense(16, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # Define decoder
    latent_inputs = Input(shape=(latent_dim,))
    h_decoder = Dense(16, activation='relu')(latent_inputs)
    outputs = Dense(input_dim)(h_decoder)

    decoder = Model(latent_inputs, outputs, name='decoder')

    # Define VAE
    outputs_vae = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs_vae, name='vae')

    # Loss function
    reconstruction_loss = mse(inputs, outputs_vae)
    reconstruction_loss *= input_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    # Metrics tracking
    start_time = time.time()
    memory_before = memory_usage()[0]

    # Train the VAE
    history = vae.fit(data, data, epochs=epochs, batch_size=batch_size, verbose=0)

    memory_after = memory_usage()[0]
    end_time = time.time()

    # Generate synthetic data
    z_sample = np.random.normal(size=(data.shape[0], latent_dim))
    generated_data = decoder.predict(z_sample)

    # Output metrics
    print("\nMetrics for VAE:")
    print("Number of samples:", data.shape[0])
    print("Number of features:", data.shape[1])
    print("Latent space dimensionality:", latent_dim)
    print("Convergence time (seconds):", end_time - start_time)
    print("Memory used (MB):", memory_after - memory_before)

    # Complexity class
    complexity_class = "O(n_samples * epochs * n_features)"
    complexity_name = "Linear Time (O(n))" if data.shape[0] * epochs * data.shape[1] < 1e6 else "Quadratic Time (O(n^2))"
    print("Complexity Class:", complexity_class)
    print("Complexity Name:", complexity_name)

    # Plot training loss
    plt.plot(history.history['loss'], label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('VAE Training Loss')
    plt.grid()
    plt.show()

    # Plot synthetic data (first 2 dimensions)
    if latent_dim == 2:
        plt.scatter(generated_data[:, 0], generated_data[:, 1], alpha=0.5)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Generated Synthetic Data (2D projection)')
        plt.grid()
        plt.show()

if __name__ == "__main__":
    vae_synthetic_data()
