# Aspect-Based Sentiment Extraction
Codes and Datasets for the bachelor thesis: 'An Unsupervised Neural Attention Model for Sentiment Classification' by Maarten de Ruiter (2020).  The code is mainly provided and inspired by He et al. (2017) and He et al. (2018). It is altered and the pre-processing scripts of HAABSA++ (Trusca et al., 2020) are adapted to connect to the other scripts.

## Data
You can find the .xml files for SemEval-2015 and SemEval-2016 in the data_aspect folder > externalData. The pre-trained GloVe embeddings can be downloaded from the links below and should be put in the glove folder.

- GloVe word embeddings (SemEval 2015): <https://drive.google.com/file/d/14Gn-gkZDuTVSOFRPNqJeQABQxu-bZ5Tu/view?usp=sharing>
- GloVe word embeddings (SemEval 2016): <https://drive.google.com/file/d/1UUUrlF_RuzQYIw_Jk_T40IyIs-fy7W92/view?usp=sharing>

## Train
Under code/ and type the following command for training:
```
python train.py -o output_dir
```
where *--domain* in ['res15', 'res16'] is the corresponding domain, and *-o* is the path of the output directory. You can find more arguments/hyper-parameters defined in train.py with default values used in the experiments.

After training, two output files will be saved in code/output_dir/domain/: 1) *aspect.log* contains extracted aspects with top 100 words for each of them. 2) *model_param* contains the saved model weights

## Evaluation
Under code/ and type the following command:
```
python evaluation.py -o output_dir
```
Note that you should keep the values of arguments for evaluation the same as those for training as we need to first rebuild the network architecture and then load the saved model weights.

This will output a file *att_weights* that contains the attention weights on all test sentences in code/output_dir/domain.

## Dependencies

See requirements.txt
You can install prerequirements in a virtual environment, using the following command.

```
pip install -r requirements.txt
```

## Software explanation:

The environment contains the following files that can be run:

- dataReader.py: program that reads the provided .xml files and delivers raw_data as output.
- evaluation.py: program that evaluates the results on a specified test set. Note that the arguments should be the same as the train.py file.
- model.py: program that passes the input through the layers of the model.
- my_layers.py: program specifying customized layers of the model.
- optimizers.py: program to call different optimizers, Adam is set as default.
- reader.py: program that reads the raw data and prepares it to be used as input for the model.
- train.py: program to train the model. Preprocess should be set to True on the first run. 
- utils.py: program containing basic functions for handling output and arguments to the console.
- w2vEmbReader.py: program that matches each word to the corresponding GloVe embedding.

## References

He,  R.,  Lee,  W.  S.,  Ng,  H.  T.,  and  Dahlmeier,  D.  (2017).   An  unsupervised  neural  attentionmodel  for  aspect  extraction.   InProceedings  of  the  55th  Annual  Meeting  of  the  Associationfor  Computational  Linguistics  (ACL  2017),  pages  388–397.  Association  for  ComputationalLinguistics.



He, R., Lee, W. S., Ng, H. T., and Dahlmeier, D. (2018).  Exploiting document knowledge for aspect-level sentiment classification.  In Gurevych,  I. and Miyao,  Y.,  editors,Proceedings  of the 56th Annual Meeting of the Association for Computational Linguistics (ACL 2018), pages579–585. Association for Computational Linguistics.



Trusca,  M.  M.,  Wassenberg,  D.,  Frasincar,  F.,  and  Dekker,  R.  (2020).   A  hybrid  approach for aspect-based sentiment analysis using deep contextual word embeddings and hierarchical attention. In 20th International Conference on Web Engineering (ICWE 2020), volume 12128 of LNCS, pages 365–380. Springer.

