import numpy as np
from NN_numpy import NeuralNet, Layer
from snake import Game, nbFeatures, nbActions
import concurrent.futures
import random

class Individu:
    def __init__(self, nn, id):
        self.nn = nn
        self.id = id
        self.score = 0

    def clone(self):
        copie = Individu(NeuralNet([layer.size for layer in self.nn.layers]), self.id)
        for idx, layer in enumerate(copie.nn.layers[1:]):
            layer.bias = self.nn.layers[idx+1].bias.copy()
            layer.weights = self.nn.layers[idx+1].weights.copy()
        return copie

def eval(sol, gameParams):
    sol.score = 0.0
    for _ in range(gameParams["nbGames"]):
        game = Game(gameParams["height"], gameParams["width"])
        steps = 0
        max_steps = gameParams.get("max_steps", 1000)  
        while game.enCours and steps < max_steps:
            pred = np.argmax(sol.nn.compute(game.getFeatures()))
            game.direction = pred
            game.refresh()
            steps += 1
        sol.score += len(game.serpent)
    return sol.id, sol.score

def optimize(taillePopulation, tailleSelection, pc, mr, arch, gameParams, nbIterations):
    population = [Individu(NeuralNet(arch), i) for i in range(taillePopulation)]
    
    for it in range(nbIterations):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            scores = list(executor.map(eval, population, [gameParams]*taillePopulation))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        bestScores = scores[:tailleSelection]

        bestIndividuals = [population[id] for id, score in bestScores]

        for i in range(tailleSelection, taillePopulation, 2):
            parent1 = random.choice(bestIndividuals)
            parent2 = random.choice(bestIndividuals)
            copie1 = parent1.clone()
            copie2 = parent2.clone()

            for layer1, layer2 in zip(copie1.nn.layers[1:], copie2.nn.layers[1:]):
                if random.random() < mr:
                    layer1.weights += np.random.normal(0, 1, layer1.weights.size).reshape(layer1.weights.shape)
                if random.random() < mr:
                    layer2.weights += np.random.normal(0, 1, layer2.weights.size).reshape(layer2.weights.shape)
            
            population[i] = copie1
            population[i+1] = copie2

    return population[0].nn

if __name__ == '__main__':
    taillePopulation = 400
    tailleSelection = 50
    pc = 0.8
    mr = 0.2
    arch = [nbFeatures, 24, nbActions]  
    gameParams = {"nbGames": 10, "height": 10, "width": 10, "max_steps": 1000}
    nbIterations = 1000

    nn = optimize(taillePopulation, tailleSelection, pc, mr, arch, gameParams, nbIterations)
    nn.save("model.txt")