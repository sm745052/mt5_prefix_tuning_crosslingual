# MT5 prefix tuning

This repository is the implementation of [Ablation Study and Experimental Analysis of Zero shot Cross-Lingual Question Answering on TyDiQA Dataset](https://drive.google.com/file/d/1FV4KMcr_JM8pI-JQ_Nymojd0KLHSi4PI/view?usp=sharing). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python final.py
```
Hyper parameters (lr, num_epochs, batch_size) can be tuned in the file itself. Add "tydiqa.en.train.json" to the main folder

## Evaluation

To evaluate my model on TydiQA different languages, run:

```eval
cd eval
python tydiqa_eval_.py > results.txt
```
For language selection, change path variable to the resective dataset.

> For best results keep learning_rate = 3e-3, decay = 0.01, num_epochs = 30, batch_size = 8

## Results

Our model achieves the following performance on :

### [TyDiQA (All languages)]
![image](https://i.ibb.co/6bp1Py7/result.png)
>The training has been done on only english data of TyDiQA train dataset. (set of 3696 English QA)
