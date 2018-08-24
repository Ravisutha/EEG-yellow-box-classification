# Epileptic transients detection in EEG signals
The objective is to detect epileptic transients in the EEG signals. The data contains eeg signals sampled at different sampling rates. The training set contains yellow box annotations to indicate the presence of the transients.

### Classification Model
Four stacked layers of LSTMs are used to train sequences of eeg signals. Detailed explanation of input window size and classification model details can found in the report [rsakrep-report_takehome2.pdf](https://github.com/Ravisutha/EEG-yellow-box-classification/blob/master/rsakrep-report_takehome2.pdf).

### Prerequisites
----
Following are some packages that are a must to run the code.
>1. [PyEDFlib](https://pyedflib.readthedocs.io/en/latest/)  - To extract signal stored in edf files.
>2. [Keras](https://keras.io/) - With tensorflow backend for building the model.


### Author
-----
* [Ravisutha Sakrepatna Srinivasamurthy](https://www.linkedin.com/in/ravisutha/)