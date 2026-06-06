import pygame

import config
from tetris import Tetris


class TetrisEnv:
    def __init__(self, render_mode = False):
        self.reset(render_mode)

    def reset(self, render_mode = False):
        self.game = Tetris(config.BOARD_WIDTH,config.BOARD_HEIGHT)
        
        if render_mode:
            pygame.init()
            screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
            pygame.display.set_caption('Tetris')

        return self.get_observation()

            


