import numpy as np
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
            self.screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
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
            
    def count_holes(self):
        total_holes = 0

        for x in range(0,9):
            found_filled = False
            for y in range(0,19):
                if self.game.grid[y][x] != 0:
                    found_filled = True
                elif found_filled:
                    total_holes += 1

        return total_holes

    def aggregate_height(self):
        total = 0

        for x in range(config.BOARD_WIDTH):
            for y in range(config.BOARD_HEIGHT):
                if self.game.grid[y][x] != 0:
                    total += (config.BOARD_HEIGHT - y)
                    break

        return total

    def compute_bumpiness(self):
        heights = []
        for x in range(config.BOARD_WIDTH):
            height = 0

            for y in range(config.BOARD_HEIGHT):
                if self.game.grid[y][x] != 0:
                    height = config.BOARD_HEIGHT - y
                    break

            heights.append(height)

        bumpiness = 0

        for i in range(len(heights) - 1):
           bumpiness += abs(heights[i]-heights[i+1])

        return bumpiness

    def compute_reward(self,lines_cleared, holes_before,holes_after,height_before,height_after,bumpiness_before,bumpiness_after,game_over):
        reward = 0

        line_rewards = [0,1,3,5,8]
        new_holes_penalty = 0.5
        height_penalty = 0.1
        bumpiness_penalty = 0.1
        game_over_penalty = -2

        new_holes = holes_after - holes_before
        height_diff = height_after - height_before
        bumpiness_diff = bumpiness_after - bumpiness_before

        reward += line_rewards[lines_cleared]
        if new_holes > 0:
            reward -= new_holes_penalty * new_holes
        if height_diff > 0:
            reward -= height_penalty * height_diff
        if bumpiness_diff > 0:
            reward -= bumpiness_penalty * bumpiness_diff
        if game_over:
            reward -= game_over_penalty

        return reward



    def step(self, action):
        placements = self.get_valid_placement()
        
        rotation, column = placements[action]
        
        # screenshot the board before placing any blocks
        holes_before = self.count_holes()
        height_before = self.aggregate_height()
        bumpiness_before = self.compute_bumpiness()
        
        # apply placement to the piece
        self.game.current_piece.rotation = rotation
        self.game.current_piece.x = column
        self.game.current_piece.y = -2
        lines_cleared = self.game.hard_drop()

        # screenshot the board after placing the blocks
        holes_after = self.count_holes()
        height_after = self.aggregate_height()
        bumpiness_after = self.compute_bumpiness()

        reward = self.compute_reward(
            lines_cleared,
            holes_before,
            holes_after,
            height_before,
            height_after,
            bumpiness_before,
            bumpiness_after,
            self.game.game_over
        )

        observation = self.get_observation()

        return observation, reward, self.game.game_over, False, {}

    # to check what action indices are valid
    def action_masks(self):
        placements = self.get_valid_placement()
        num_valid = len(placements)

        array = np.zeros(config.MAX_PLACEMENTS, dtype=bool)

        for i in range(num_valid):
            array[i] = True

        return array

    def get_observation(self):
        flat_arr = [item for row in self.game.grid for item in row]
        numpy_flat_arr = np.array(flat_arr)

        return numpy_flat_arr




        


