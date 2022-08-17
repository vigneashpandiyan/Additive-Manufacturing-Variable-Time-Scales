# Additive-Manufacturing-Variable-Time-Scales

This repo hosts the codes that were used in journal work "Deep learning-based monitoring of laser powder bed fusion process on variable time-scales using heterogeneous sensing and operando X-ray radiography guidance".

![Experimental setup](https://user-images.githubusercontent.com/39007209/185176561-efe2dbff-ab0d-4791-bdfa-4b267eb3b875.jpg)

# Journal link
https://doi.org/10.1016/j.addma.2022.103007

# Overview

The recent advances in sensorization and processing of the associated process zone signatures using Machine Learning techniques have made qualifying parts on the fly. However, the in-situ monitoring strategies based on classifying processing regimes reported in the literature operate on signals of fixed length in time, constraining the generalization of the trained ML model by not allowing monitoring processes with heterogeneous laser scanning strategies. In the article recently published in collaboration with EPFL (École polytechnique fédérale de Lausanne) and Paul Scherrer Institut PSI in "Additive Manufacturing Journal [impact factor 11.6]", we try to bridge this gap by developing a hybrid Deep Learning (DL) model by combining Convolutional Neural Networks with Long-Short Term Memory that can operate over variable time-scales. During the validation procedure of the trained hybrid DL model, the model could operate on time scales ranging from 0.5 ms to 4 ms, thereby opening an opportunity in real time monitoring of build jobs involving complex tool paths in metal-Additive Manufacturing. The sensitivity analysis of the trained hybrid model also showed that the optical emissions in laser wavelength and acoustic emission signatures carried more relevant information to guide the decision-making process.

![Additive Manufacturing Graphical Abstract](https://user-images.githubusercontent.com/39007209/185176347-701a574a-6e5d-43b5-81e6-8b7f8e87a9f5.jpg)

# Variable time-scales

The major drawback of neural networks based on CNN is that they only accept input data with a fixed size, and they process them all at once to produce a fixed amount of output data each time. This processing scheme implies that they cannot be employed for data with different lengths. Unlike CNN's, Recurrent Neural Networks (RNNs) do not process all the input data simultaneously. Instead, they process the input data one data point (the smallest unit in which a signal can be divided) at a time, treating the input signal as a sequence. Indeed, the RNN performs its computation on the first element of the input sequence before producing an output. The output, known as the hidden state, is then combined with the following input in the sequence to produce another output. This computation continues until the model encounters all the elements in the sequence so that the final output is dependent on all the sequence's elements. The computational unit that performs the operations on the current sequence's element and hidden state is called the RNN cell, and it is reused at each time step. This mechanism enables RNNs to exploit dynamically changing temporal information from the input sequences for decision-making. There are many variants of RNNs such as Vanilla RNNs, Gated Recurrent Units (GRU’s), LSTM, and Bi-directional LSTMs. 
	
Given the CNNs ability to find patterns in the input data and RNNs capacity to discover temporal relationships regardless of the sequence duration, combining both help develop hybrid DL models with interesting properties. The combination of such networks has been applied to a variety of tasks such as forecasting, classification , sentiment analysis. As far as this work is concerned, we have built a DL architecture combining a CNN and an LSTM block, namely CNN-LSTM, a network that can flexibly operate over variable time scales. LSTM network was chosen over other variants of RNN as they are capable of learning very long order dependencies. The proposed hybrid DL model is schematized in Figure below. As can be seen, CNN acts as the front-end for the proposed model by processing the input data to extract features out of them. As CNNs preserve the signal structure, the processed data is flattened (converted into a vector) before feeding the RNN. The RNN then learns the temporal relationship in the data irrespective of the vector size and performs the decision-making task by outputting a class. Notice that, by combining a CNN (susceptible to the input data size) and an RNN (not affected by the input size), we were able to achieve one of our goal, which is to have an ML model that can predict on inputs with variable time-scales.

![1-s2 0-S2214860422004006-gr1_lrg](https://user-images.githubusercontent.com/39007209/185177781-e8015896-2f3e-44f3-80b6-0049b3452aa1.jpg)

# Code
```bash
git clone https://github.com/vigneashpandiyan/Additive-Manufacturing-Variable-Time-Scales
cd Additive-Manufacturing-Variable-Time-Scales

```

# Citation
```
@article{pandiyan2022deep,
  title={Deep learning-based monitoring of laser powder bed fusion process on variable time-scales using heterogeneous sensing and operando X-ray radiography guidance},
  author={Pandiyan, Vigneashwara and Masinelli, Giulio and Claire, Navarre and Le-Quang, Tri and Hamidi-Nasab, Milad and de Formanoir, Charlotte and Esmaeilzadeh, Reza and Goel, Sneha and Marone, Federica and Log{\'e}, Roland and others},
  journal={Additive Manufacturing},
  volume={58},
  pages={103007},
  year={2022},
  publisher={Elsevier}
}
```

