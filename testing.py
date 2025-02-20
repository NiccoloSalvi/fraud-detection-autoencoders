import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import jax.numpy as jnp
import jax
import jax.nn as nn
import jax.scipy.optimize
import matplotlib.pyplot as plt

data = pd.read_csv('creditcard.csv')
dataset = data.to_numpy()

# initialize neural network for encoder
n1, n2, n3, n4, n5 = 31, 23, 19, 17, 8
np.random.seed(0) # for reproducibility
# creates a matrix  n2xn1 of samples from normal distribution
# where for every node in n2 there is a weight for the corresponding layer in n1
W1 = np.random.randn(n2, n1)
b1 = np.zeros((n2, 1))
W2 = np.random.randn(n3, n2)
b2 = np.zeros((n3, 1))
W3 = np.random.randn(n4, n3)
b3 = np.zeros((n4, 1))
W4 = np.random.randn(n5, n4)
b4 = np.zeros((n5, 1))
params_encoder = [W1, b1, W2, b2, W3, b3, W4, b4]
params_encoder = [jnp.array(p) for p in params_encoder]

def encoder(x, params, is_training=True):
  p1 = 0.1
  p2 = 0.2
  # unfold the parameters
  W1, b1, W2, b2, W3, b3, W4, b4 = params

  # first layer is x
  layer1 = x.T
  # other layers use the activation function
  layer2 = nn.relu(W1 @ layer1 + b1)
  if is_training:
    dropout_mask = (np.random.rand(*layer2.shape) > p1) / (1 - p1)
    layer2 *= dropout_mask

  layer3 = nn.relu(W2 @ layer2 + b2)
  if is_training:
    dropout_mask = (np.random.rand(*layer3.shape) > p2) / (1 - p2)
    layer3 *= dropout_mask

  layer4 = nn.relu(W3 @ layer3 + b3)
  # output
  layer5 = nn.relu(W4 @ layer4 + b4)

  return layer5.T

# initialize before going on
output = encoder(jnp.array(dataset), params_encoder, is_training=False)

n1, n2, n3, n4, n5 = 8, 17, 19, 23, 31
W1 = np.random.randn(n2, n1)
b1 = np.zeros((n2, 1))
W2 = np.random.randn(n3, n2)
b2 = np.zeros((n3, 1))
W3 = np.random.randn(n4, n3)
b3 = np.zeros((n4, 1))
W4 = np.random.randn(n5, n4)
b4 = np.zeros((n5, 1))

params_decoder = [W1, b1, W2, b2, W3, b3, W4, b4]
params_decoder = [jnp.array(p) for p in params_decoder]

def decoder(x, params, is_training=True):
  p1 = 0.2
  p2 = 0.1
  # unfold the parameters
  W1, b1, W2, b2, W3, b3, W4, b4 = params

  # first layer is x
  layer1 = x.T
  # other layers use the activation function
  layer2 = nn.relu(W1 @ layer1 + b1)
  if is_training:
    dropout_mask = (np.random.rand(*layer2.shape) > p1) / (1 - p1)
    layer2 *= dropout_mask

  layer3 = nn.relu(W2 @ layer2 + b2)
  if is_training:
    dropout_mask = (np.random.rand(*layer3.shape) > p2) / (1 - p2)
    layer3 *= dropout_mask

  layer4 = nn.relu(W3 @ layer3 + b3)

  # output
  layer5 = nn.sigmoid(W4 @ layer4 + b4)

  return layer5.T

decoder(jnp.array(output), params_decoder, is_training=False)

# Use jnp.array to ensure compatibility with JAX
small_dataset = jnp.array(dataset[:1000])
# small_dataset = jnp.array(dataset)

def gradientDescent():
    history_loss = []

    for epoch in range(num_epochs):
        # Compute gradients for the entire batch
        grad_encoder, grad_decoder = jax.grad(loss_fn, argnums=(0, 1))(params_encoder, params_decoder, small_dataset)
        
        # Update weights
        params_encoder = [p - learning_rate * g for p, g in zip(params_encoder, grad_encoder)]
        params_decoder = [p - learning_rate * g for p, g in zip(params_decoder, grad_decoder)]
        
        current_loss = loss_fn(params_encoder, params_decoder, small_dataset)
        history_loss.append(current_loss)
        if epoch % 50 == 0:
            current_loss = loss_fn(params_encoder, params_decoder, small_dataset)
            print(f"Epoch {epoch}, Loss: {current_loss}")

    # plot the history loss
    plt.figure()
    plt.plot(history_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

def loss_fn(params_encoder, params_decoder, batch):
    encoded = encoder(batch, params_encoder)
    decoded = decoder(encoded, params_decoder)
    return jnp.min(jnp.mean((batch - decoded) ** 2))

history_loss = []

num_epochs = 1000
learning_rate = 0.01
batch_size = 128

scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), unit_variance=False)
dataset[:, [0, 29]] = scaler.fit_transform(dataset[:, [0, 29]])

# Use jnp.array to ensure compatibility with JAX
small_dataset = jnp.array(dataset)

key = jax.random.PRNGKey(0)
for epoch in range(num_epochs):
    key, subbkey = jax.random.split(key)
    perm = jax.random.permutation(subbkey, small_dataset.shape[0])
    for i in range(0, small_dataset.shape[0], batch_size):
        batch_idx = perm[i : i+batch_size]
        data_batch = small_dataset[batch_idx]

        # Compute gradients for the entire batch
        grad_encoder, grad_decoder = jax.grad(loss_fn, argnums=(0, 1))(params_encoder, params_decoder, data_batch)
        
        # Update weights
        params_encoder = [p - learning_rate * g for p, g in zip(params_encoder, grad_encoder)]
        params_decoder = [p - learning_rate * g for p, g in zip(params_decoder, grad_decoder)]
        
    current_loss = loss_fn(params_encoder, params_decoder, small_dataset)
    history_loss.append(current_loss)
    if epoch % 50 == 0:
        current_loss = loss_fn(params_encoder, params_decoder, small_dataset)
        print(f"Epoch {epoch}, Loss: {current_loss}")

# plot the history loss
plt.figure()
plt.plot(history_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()