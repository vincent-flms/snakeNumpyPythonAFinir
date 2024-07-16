from snake import Game, nbFeatures, nbActions 
from vue import SnakeVue
import genetic
import pygame
import numpy as np  

if __name__ == '__main__':
    gameParams = {"nbGames": 10, "height": 10, "width": 10}
    arch = [nbFeatures, 24, nbActions]  
    nbIterations = 1000

    
    nn = genetic.optimize(taillePopulation=400, tailleSelection=50, pc=0.8, mr=0.2, arch=arch, gameParams=gameParams, nbIterations=nbIterations)
    nn.save("model.txt")

    
    pygame.init()
    vue = SnakeVue(gameParams["width"], gameParams["height"], 64)
    fps = pygame.time.Clock()
    gameSpeed = 20

    while True:
        game = Game(gameParams["height"], gameParams["width"])
        while game.enCours:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
        
            pred = np.argmax(nn.compute(game.getFeatures()))
            game.direction = pred
            game.refresh()

            vue.displayGame(game)
            fps.tick(gameSpeed)

        pygame.time.wait(1000)