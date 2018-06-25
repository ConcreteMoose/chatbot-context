# chatbot-context

Both the encoder-decoder model and exhaustive search model can be found here. The dataset is not added, since it is too big for github. 

The code to generate the wordpiece model can be found in the folder Wordpiece model. The json file it generates is also added to both chatbot models.

### Encoder-Decoder model with attention

The encoder decoder model can be found in the Encoder-Decoder folder. The code is found in the main folder. `seq2seq.py` trains the model, and `telegram.py` runs the model on the telegram-bot we have set up. `attention_decoder.py` and `tdd.py` are external code to add attention to our model. They can be found [here](https://github.com/datalogue/keras-attention). 
In the resources folder, the trained model weights are found, and the wordpiece model is found. We did not include our dataset due to its size.

### TF-IDF model

Exhaustive search using TF-IDF model. For convenience, python files are numbered in the order they should be run. Pre-processing steps from the folder 'preprocessing' should be run first. The chatbot performing exhaustive search is also included in that folder. The preprocessing and chatbot for the context-based chatbot are located in the 'clustering' folder. All data and representations of processed data are stored in the 'res' folder.
