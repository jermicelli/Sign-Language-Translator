from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import onnx
import onnxruntime as ort

from DatasetPrepare import dataLoaders
from DatasetNNTrain import NeuralNetwork


def evaluate(outputs: Variable, labels: Variable) -> float:
    Y = labels.numpy()
    Yhat = np.argmax(outputs, axis=1)
    return float(np.sum(Yhat == Y))


def batch_evaluate(model: NeuralNetwork, dataloader: torch.utils.data.DataLoader) -> float:
    score = n = 0.0
    for batch in dataloader:
        n += len(batch['image'])
        outputs = model(batch['image'])
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().numpy()
        score += evaluate(outputs, batch['label'][:, 0])
    return score / n

def validate():
    train_loader, test_loader = dataLoaders()
    model = NeuralNetwork().float().eval()
    pretrainedModel = torch.load("D:/Projects/MachineLearningASL/src/DatasetNNTrain/DatasetNNTrain/HandModel.pth")
    model.load_state_dict(pretrainedModel)
    print('=' * 10, 'PyTorch', '=' * 10)
    train_acc = batch_evaluate(model, train_loader) * 100.
    print('Training accuracy: %.1f' % train_acc)
    test_acc = batch_evaluate(model, test_loader) * 100.
    print('Validation accuracy: %.1f' % test_acc)

    train_loader,test_loader = dataLoaders(1)
    fname = "ASLWebcam.onnx"
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(model, dummy, fname, input_names=['input'])
    onnxmodel = onnx.load(fname)
    onnx.checker.check_model(onnxmodel)
    ort_session = ort.InferenceSession(fname)
    model = lambda inp: ort_session.run(None, {'input': inp.data.numpy()})[0]

    print('=' * 10, 'ONNX', '=' * 10)
    train_acc = batch_evaluate(model, train_loader) * 100.
    print('Training accuracy: %.1f' % train_acc)
    test_acc = batch_evaluate(model, test_loader) * 100.
    print('Validation accuracy: %.1f' % test_acc)

if __name__ == '__main__':
    validate()