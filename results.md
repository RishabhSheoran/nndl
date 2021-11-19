# Evaluation and Interpretation

| Model                   | Parameters | Error % |
| ----------------------- | ---------- | ------- |
| MLP - baseline          | 55.30M     | 32.46   |
| MLP - with improvements | 55.50M     | 30.72   |
| CNN - baseline          | 2.77M      | 27.06   |
| CNN - with improvements | 23.09M     | 27.01   |
| RNN - baseline          | 138K       | 29.16   |
| RNN - with improvements | 229K       | 32.83   |
| ANN - baseline          | 4.73M      | 25.45   |
| ANN - with improvements | 29.07M     | 25.45   |

The above table shows the summary of the results across all networks. As seen earlier, error rates of ANNs are deceptive as the predictions from the network are sub-par. Hence, we can safely disregard ANN as candidate for our task. Among those that are left, CNNs give the best performance, purely based on the error rate statistic. Although it is mighty impressive that RNN could almost match the performance of CNNs using only a fraction of the parameters.

RNN is a suitable candidate until we consider the application of the task at hand. We want the models to be extended for videos and for such a scale, the sequential nature of RNNs become performance bottlenecks. For this reason, we recommed using CNN for application of deep neural networks in multi-label classification problem.
