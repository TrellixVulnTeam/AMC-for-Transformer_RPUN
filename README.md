# AMCPruningForTransformer 19205945 Pan Feng
1)Preparing dataset
python3 -m spacy download en
python3 -m spacy download de
python3 preprocess.py -lang_src de -lang_trg en -share_vocab -save_data m30k_deen_shr.pkl

2)Search space by following code:
python3 Search.py --model_type transformer --ckpt model/model.chkpt

3)create a nni experiment by
nnictl create -c config.yml -port 8080
where config.yml is my searching strategy
compressed model will be stored in /output
