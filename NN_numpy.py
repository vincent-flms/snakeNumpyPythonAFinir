import numpy as np
import random
import math

class Layer:
    def __init__(self, size, previousSize):
        self.aggregations = np.zeros(size, dtype=np.float64)
        self.outputs = np.zeros(size, dtype=np.float64)
        self.size = size
        self.bias = np.array([random.uniform(-0.3, 0.3) for _ in range(size)], dtype=np.float64)
        self.weights = np.array([np.array([random.uniform(-1.0/np.sqrt(previousSize), 1.0/np.sqrt(previousSize)) for _ in range(previousSize)], dtype=np.float64) for _ in range(size)])

    def compute(self, previousValues):
        self.aggregations = np.dot(previousValues, self.weights.T) + self.bias
        self.outputs = 1.0 / (1.0 + np.exp(-self.aggregations))

class NeuralNet:
    def __init__(self, layerSizes):
        self.layers = [Layer(layerSizes[i], layerSizes[i-1] if i >= 1 else 0) for i in range(len(layerSizes))]
        self.architecture = layerSizes

    def getVector(self):
        res = np.array([])
        for layer in self.layers[1:]:
            res = np.concatenate((res, layer.bias))
            res = np.concatenate((res, layer.weights.flatten()))
        return res

    def compute(self, features):
        self.layers[0].outputs = features
        for i in range(1, len(self.layers)):
            self.layers[i].compute(self.layers[i-1].outputs)
        return self.layers[-1].outputs

    def load(self, filename):
        with open(filename, "r") as file:
            lines = file.readlines()
            file.close()
            layerSizes = [int(size) for size in lines[0][:-1].split(" ")]
            self.layers = [Layer(layerSizes[i], layerSizes[i-1] if i >= 1 else 0) for i in range(len(layerSizes))]
            idx = 1
            for layer in self.layers[1:]:
                layer.bias = np.array([float(size) for size in lines[idx][:-1].split(" ")])
                idx += 1
                for i in range(layer.size):
                    layer.weights[i] = np.array([float(size) for size in lines[idx][:-1].split(" ")])
                    idx += 1

    def save(self, filename):
        with open(filename, "w") as file:
            file.write(" ".join([str(layer.size) for layer in self.layers]) + "\n")
            for layer in self.layers[1:]:
                file.write(" ".join([str(bias) for bias in layer.bias]) + "\n")
                for weights in layer.weights:
                    file.write(" ".join([str(weight) for weight in weights]) + "\n")