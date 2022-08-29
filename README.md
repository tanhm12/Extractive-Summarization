# English and Vietnamese Extractive Summarization with small BERT model
## English version
 - BERT-small + CNN Encoder + biLSTM Encoder-Decoder with R1/R2/RougeL : 30.94/12.50/27.58 on cnn (of cnn/dailymail) dataset
 - Much faster and more efficient in terms of resources than BERT.
 - Example inference scripts are in **inference.ipynb**
 - Trained model for CNN dataset [here](https://drive.google.com/drive/folders/1CGFl9ihei9jSqoT6ITXwkU47ri9bXL1J?usp=sharing) and should be saved to "save/cnn/best-model"
## Vietnamese version
 - Small version of Bert (l6-h384) + CNN Encoder + biLSTM Encoder-Decoder with R1/R2/RougeL : 56.29/24.27/35.70 on [Vietnews](https://github.com/ThanhChinhBK/vietnews) dataset
 - Much faster and more in terms of resources than BERT.
 - Example inference scripts are in **inference.ipynb**
 - Trained model for Vietnews dataset [here](https://drive.google.com/drive/folders/1BzGfX-7V-Q2B-jNn0bPqq7-zzPqplw1v?usp=sharing) and should be saved to "save/vietnews/best-model"
