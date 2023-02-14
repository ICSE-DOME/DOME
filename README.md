# ICSE22_DOME
This is a replication package for ICSE23 Paper `Developer-Intent Driven Code Comment Generation`.<br>

## Content
1. [Project Summary](#1-Project-Summary)<br>
2. [Get Started](#2-Get-Started)<br>
&ensp;&ensp;[2.1 Requirements](#21-Requirements)<br>
&ensp;&ensp;[2.2 Dataset](#22-Dataset)<br>
&ensp;&ensp;[2.3 Comment Intent Labeling Tool](#23-Comment-Intent-Labeling-Tool)<br>
&ensp;&ensp;[2.4 Developer-Intent Driven Comment Generator](#24-Developer-Intent-Driven-Comment-Generator)<br>
3. [Application](#3-Application)<br>

## 1 Project Summary
Existing automatic code comment generators mainly focus on producing a general description of functionality for a given code snippet without considering developer intentions. However, in real-world practice, comments are complicated, which often contain information reflecting various intentions of developers, e.g., functionality summarization, design rationale, implementation details, code properties, etc. To bridge the gap between automatic code comment generation and real-world comment practice, we define Developer-Intent Driven Code Comment Generation, which can generate intent-aware comments for the same source code with different intents. To tackle this challenging task, we propose DOME, an approach that utilizes Intent-guided Selective Attention to explicitly select intent-relevant information from the source code, and produces various comments reflecting different intents. Our approach is evaluated on two real-world Java datasets, and the experimental results show that our approach outperforms the state-of-the-art baselines. A human evaluation also confirms the significant potential of applying DOME in practical usage, enabling developers to comment code effectively according to their own needs.

## 2 Get Started
### 2.1 Requirements
* Hardwares: NVIDIA GeForce RTX 3060 GPU, intel core i5 CPU
* OS: Ubuntu 20.04
* Packages: 
  * python 3.9
  * pytorch 1.9.0
  * cuda 11.1
  * transformers 4.9.2
  * numpy
  * tqdm

### 2.2 Dataset
#### 2.2.1 benchmark datasets
*DOME* is evaluated on [Funcom](http://leclair.tech/data/funcom/) and [TLC](https://github.com/xing-hu/TL-CodeSum) benchmark datasets.<br>
[Shi et al.](https://arxiv.org/pdf/2207.05579) have reported that many benchmark datasets have noisy data, e.g. *Verbose Sentence*, *Content Tampering*. So we directly use the "clean" version of the two datasets open sourced by them (https://github.com/BuiltOntheRock/FSE22_BuiltOntheRock).
#### 2.2.2 manually labeled dataset
Since training and evaluating *DOME* require a large volume of labeled comment-intent data, we develop a COmment-INtent labeling tool, named *COIN*, to support the automatic annotation of comment intents for the codecomment dataset. To train *COIN*, we randomly sample 20K code-comment pairs from Funcom and TLC (10K data for each), and invite five developers to manually classify the data into six intent categories. The manually labeled dataset can be found in ```src/comment_classifier/dataset/manually_labeled_data_20000.xlsx```.
#### 2.2.3 auto-labeled dataset
Then, we use the well-trained *COIN* to automatically annotate the comments in the two datasets with the corresponding intent categories. Since the others comments are seen as unspecified or ambiguous
comments, we exclude all data with the intent category of others. In common with [Rencos](https://github.com/zhangj111/rencos), we further remove the exactly duplicated code-comment pairs in the test set for TLC dataset. The preprocessed datasets can be downloaded at [DOME_Dataset](https://drive.google.com/file/d/1KBJysjgJ1i6UDB5O--44Z1YT9u-qCZiG/view?usp=sharing)


### 2.3 Comment Intent Labeling Tool
1. Download the pretrained CodeBERT model from [transformers](https://huggingface.co/microsoft/codebert-base) and put the model file into the ```src/comment_classifier/pretrained_codebert``` directory.

2. Train the Comment-Intent Labeling Tool *COIN* model with intent-labeled dataset.
```
python src/comment_classifier/train.py
```
3. Use the well-trained *COIN* to annotate the two benchmark datasets automatically.
```
python src/comment_classifier/prediction.py
```

### 2.4 Developer-Intent Driven Comment Generator
1. Go the ```src/comment_generator``` directory, download the preprocessed datasets and put them into the ```dataset``` directory, and generate the vocabulary:
```
cd src/comment_generator
python ./dataset/{DATASET}/generate_vocab.py
```
2. Train *DOME* model:
```
python train_DOME.py
```
3. Test *DOME* model:
```
python prediction_DOME.py
```

## 3 Application
### Using COIN to identify the developer-intent for a given comment
```
>>> classifier = commentClassifier('./comment_classifier/pretrained_codebert', 6, 0.2)
>>> classifier.load_state_dict(torch.load("./comment_classifier/saved_model/comment_classifier.pkl"))
>>> classifier.cuda()

>>> comment = 'Starts the background initialization'
>>> tokenizer = AutoTokenizer.from_pretrained('./comment_classifier/pretrained_codebert')
>>> logits = classifier(coin_preprocess(tokenizer, comment))
>>> intent = class_name[int(torch.argmax(logits, 1))]
>>> print('comment:', comment, '\nintent:', intent)
comment: Starts the background initialization 
intent: what
```
### Using DOME to generate various comments that are coherent with the given intents
```
>>> generator = Generator(config.d_model, config.d_intent, config.d_ff, config.head_num, config.enc_layer_num, config.dec_layer_num, config.vocab_size, config.max_comment_len, config.clip_dist_code, config.eos_token, config.intent_num, config.stat_k, config.token_k, config.dropout, None)
>>> generator.load_state_dict(torch.load(f"./src/comment_generator/saved_model/tlcodesum/comment_generator.pkl"))
>>> generator.cuda()

>>> for i in range(3):
...    print("code:\n", raw_code[i])
...    print("what:", prediction(input_code[i], input_exemplar[i], 'what', code_valid_len[i]))
...    print("why:", prediction(input_code[i], input_exemplar[i], 'why', code_valid_len[i]))
...    print("how-it-is-done:", prediction(input_code[i], input_exemplar[i], 'done', code_valid_len[i]))
...    print("usage:", prediction(input_code[i], input_exemplar[i], 'usage', code_valid_len[i]))
...    print("property:", prediction(input_code[i], input_exemplar[i], 'property', code_valid_len[i]))
...    print("=============================================================================")

code:
 public int hashCode(){
  return value.hashCode();
}
what: generates a hash code .
why: generates code for this object .
how-it-is-done: a method that generates a hashcode based on the contents of the string representations .
usage: this method is used when this class is used as the code .
property: return a hashcode for this text attribute .
=============================================================================

code:
 protected void writeQualifiedName(String nsAlias,String name) throws IOException {
  if (nsAlias != null && nsAlias.length() > 0) {
    writer.write(nsAlias);
    writer.write(':');
  }
  writer.write(name);
}
what: writes a qualified name to a file .
why: writes the beginning of the generated name to the given alias .
how-it-is-done: copy a qualified name , using the given class .
usage: below method will be used to write the idex file
property: returns a managed name path holding the value of the specified string .
=============================================================================

code:
 <T>List<T> onFind(Class<T> modelClass,String[] columns,String[] conditions,String orderBy,String limit,boolean isEager){
  BaseUtility.checkConditionsCorrect(conditions);
  List<T> dataList=query(modelClass,columns,getWhereClause(conditions),getWhereArgs(conditions),null,null,orderBy,limit,getForeignKeyAssociations(modelClass.getName(),isEager));
  return dataList;
}
what: handles the native query of the given table .
why: the open interface for other classes in crud package to query the first record in a table .
how-it-is-done: finds genericvalues by the conditions specified in the backwards conditions object , the where clause is the product of each of the type .
usage: method modified just override here before executing your test cases .
property: returns the order of the order specified
=============================================================================
```

For more detailed usage and examples, please refer to the [DOME_Application.ipynb](https://github.com/ICSE-DOME/DOME/blob/master/DOME_Application.ipynb).
