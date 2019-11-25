# RVAEBFA
The curse of dimensionality is a fundamental difficulty in anomaly detection for high dimensional data. In this rep, we implement a novel anomaly detection method named RVAE-BFA (Robust Variational Autoencoder with Balanced Feature Adaptation for high dimensional data anomaly detection), which significantly improves the anomaly detection performance when training data is contaminated. Rather than only utilize reconstruction error, we take Variational Autoencoder generated low dimensional embeddings into consideration. Meanwhile, we also implement a BFA(Balanced Feature Adaptation) mechanism to balance the weights of low dimensional embeddings and reconstruction errors. 

# RAAEBFA
We adopt the adversarial training criterion to perform variational inference by the adversarial network named RAAE-BFA (Robust adversarial autoencoder with Balanced Feature Adaptation for high dimensional data anomaly detection) in which we can generate extra samples when training data is not enough.

# More Information will be updated in the near future...