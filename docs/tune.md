# HyperParameter Tuning

Parameters of networks (LSTM NN & ! Dimensional Conv NN) are tuned via Keras Turner Library. 
However, `batch_size` and `epoch` are tuned individually.


## Tuning `epoch` and `batch_size`

`epoch` hyperparameters are sorting as ascending and `batch_size` hyperparameters are sorting as descending.
Each iteration sorted parameters are used and loss values are calculated.
We aim here to capture the best of the minimum `epoch` and the best of the maximum `batch_size`.

![parameter_tuning](https://user-images.githubusercontent.com/26736844/118011611-e9722d00-b358-11eb-8b02-f1d12d390a5b.png)
