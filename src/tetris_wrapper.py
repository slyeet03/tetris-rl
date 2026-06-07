import pygame

import config
from tetris import Tetris, Tetromino


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

    def get_valid_placement(self):
        placements = []
        piece = self.game.current_piece
        num_rotation = len(piece.shape)

        for rotation_idx in range(num_rotation):
            # current rotation
            shape = piece.shape[rotation_idx]

            # getting all occupied cells
            occupied = [
                (i,j)
                for i, row in enumerate (shape)
                for j, cell in enumerate (row) 
                if cell == '0'
            ]

            # j stores a range or columns occupied by the 0 string
            leftmost = min(j for _,j in occupied)
            rightmost = max(j for _,j in occupied)

            # a piece is a 5x5 grid so gotta do some shifting for it to only occupy the cells that the 0 is on
            for x_pos in range(-leftmost, self.game.width - rightmost):
                temp = Tetromino(x_pos, -2, piece.shape)
                temp.rotation = rotation_idx

                # can a piece even spawn here?
                if not self.game.valid_move(temp,0,0,0):
                    continue

                # hard drop
                while self.game.valid_move(temp,0,1,0):
                    temp.y += 1

                placements.append((rotation_idx,x_pos))

        return placements
            
            

        


