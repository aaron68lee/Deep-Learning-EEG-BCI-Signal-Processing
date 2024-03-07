# ECE147-Research Project

Our research concerns EEG signal processing for human brain waves using CNN and RNN models for action classification. 
We compare performance of CNNs and RNNs to LSTMs using F1 score as a metric for classifcation quality.

Research Paper: [EEG Decoding and Classification with CNN-LSTMs](https://docs.google.com/document/d/1aJmhZyo0kQp_02A5t0qFZwQKrVn9tUkNpD6fcJEKmU8/edit?usp=sharing)

# The Dataset

[EEG Signal][/Gallery/EEG.png]

# Model Architectures

We propose several model architectures including an vanilla CNN, LSTM, and CNN-LSTM fusion model described more in depth within our paper. 

<hr>
CNN Model<br>
[CNN Model Architecture](/Gallery/CNN_architecture.png)

LSTM Model<br>
[LSTM Model Architecture](/Gallery/LSTM_architecture.png)

ConvLSTM<br>
[CNN-LSTM Model Architecture](/Gallery/LSTM_architecture.png)

EEG-ResNet<br>
[EEG-Resnet-15 Model Architecture](/Gallery/Resnet-15_architecture.png)

<hr>
# Future Directions

* We implement EEG signal generation using a GAN architecture and assess artificial EEG generation performance using our existing classification models. 
