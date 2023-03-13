"""
这个文件是负责模型的训练的
"""
from torch.utils.data import DataLoader
import torch
from dataset import NerDataSet,collate_fn
from Model import BertNer
from train import ner_train, ner_test,ner_eval
import config

if __name__ == '__main__':
    model = BertNer()
    model.load_state_dict(torch.load(config.save_path))
    val_accuracy = -1
    train_data_set = NerDataSet("train")
    dev_data_set = NerDataSet("dev")
    train_data_loader = DataLoader(dataset=train_data_set, shuffle=True, batch_size=config.batch_size,
                                   collate_fn=collate_fn)
    val_data_loader = DataLoader(dataset=dev_data_set, shuffle=True, batch_size=config.batch_size,
                                 collate_fn=collate_fn)
    for epoch in range(config.epoches):
        ner_train(model=model, train_data_loader=train_data_loader, epoch=epoch)
        accuracy = ner_test(model=model, data_loader=val_data_loader, epoch=epoch)
        if accuracy > val_accuracy:
            torch.save(model.state_dict(), config.save_path)
            print("模型已保存。")
            val_accuracy = accuracy

    mymodel=model.load_state_dict(torch.load(config.save_path))
    ner_eval(model=model, data_loader=val_data_loader)
