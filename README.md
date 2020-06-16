
# bilstm

# bilstmTrain:
```
python3 bilstmTrain.py <a|b|c|d> <pos/ner> <path to save model> <eval num> <lr>
```

The trainFile should be in a folder called `pos` or `ner`.

The code will read to use the train and dev files that in the folder and adjust the separator
the `<eval num>` and `<lr>` are optional:
   * eval num is how many sentences to print 
   * lr is the learning rate

Example:
```
python3 bilstmTrain.py b pos model 4000 0.007
```

# bilstmPredict:
```
python3 bilstmPredict.py <a|b|c|d> <path to load the model from> <input file>
```
unlike in the bilstmTrain, the input file is the path of the file to read

