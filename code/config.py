import platform
import os
import torch



bert_path = "minirbt-h256"
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-5
ner_classes_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
batch_size = 8
save_path = "contractNerEntity.pth"
max_length = 128
last_state_dim =256
dropout = 0.5
epoches = 5
bert_path = bert_path
device = device
txt2label = {text: index for index, text in enumerate(ner_classes_list)}
label2txt = {index: text for index, text in enumerate(ner_classes_list)}
