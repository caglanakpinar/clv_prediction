# HyperParameter Tuning

Network parameters (LSTM NN & 1-Dimensional Conv NN) are tuned via the Keras Tuner library. 
However, `batch_size` and `epoch` are tuned separately.


## Tuning `epoch` and `batch_size`

`epoch` values are sorted ascending and `batch_size` values are sorted descending.
Each iteration uses the next sorted parameter pair, and loss values are calculated.
The goal here is to find the best result with the smallest `epoch` and the largest `batch_size`.

![parameter_tuning](https://user-images.githubusercontent.com/26736844/118011611-e9722d00-b358-11eb-8b02-f1d12d390a5b.png)
