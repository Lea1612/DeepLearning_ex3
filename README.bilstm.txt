Lea Setruk 
Yoel Benabou

In order to run the following command lines, you must have a following tree:
* data
    * pos
        * dev
        * train
        * test
    * ner
        * dev
        * train
        * test

To run the bilstmTrain.py file:

model A - ner
python bilstmTrain.py a data/ner/train modelA_ner ner data/ner/dev targetA_ner wordDictA_ner

model A - pos
python bilstmTrain.py a data/pos/train modelA_pos pos data/pos/dev targetA_pos wordDictA_pos

model B - ner
python bilstmTrain.py b data/ner/train modelB_ner ner data/ner/dev targetB_ner wordDictB_ner

model B - pos
python bilstmTrain.py b data/pos/train modelB_pos pos data/pos/dev targetB_pos wordDictB_pos

model C - ner
python bilstmTrain.py c data/ner/train modelC_ner ner data/ner/dev targetC_ner wordDictC_ner

model C - pos
python bilstmTrain.py c data/pos/train modelC_pos pos data/pos/dev targetC_pos wordDictC_pos

model D - ner
python bilstmTrain.py d data/ner/train modelD_ner ner data/ner/dev targetD_ner wordDictD_ner

model D - pos
python bilstmTrain.py d data/pos/train modelD_pos pos data/pos/dev targetD_pos wordDictD_pos

----------------------------------------------------------------------------------------------------------------------------------------

To run the bilstmPredict.py file:
*Note that you have first to run the bilstmTrain.py file before the bilstmPredict.py file

model A - ner
python bilstmPredict.py a modelA_ner data/ner/test ner targetA_ner.json wordDictA_ner.json

model A - pos
python bilstmPredict.py a modelA_pos data/pos/test pos targetA_pos.json wordDictA_pos.json

model B - ner
python bilstmPredict.py b modelB_ner data/ner/test ner targetB_ner.json wordDictB_ner.json

model B - pos
python bilstmPredict.py b modelB_pos data/pos/test pos targetB_pos.json wordDictB_pos.json

model C - ner
python bilstmPredict.py c modelC_ner data/ner/test ner targetC_ner.json wordDictC_ner.json

model C - pos
python bilstmPredict.py c modelC_pos data/pos/test pos targetC_pos.json wordDictC_pos.json

model D - ner
python bilstmPredict.py d modelD_ner data/ner/test ner targetD_ner.json wordDictD_ner_1.json wordDictD_ner_2.json

model D - pos
python bilstmPredict.py d modelD_pos data/pos/test pos targetD_pos.json wordDictD_pos_1.json wordDictD_pos_2.json