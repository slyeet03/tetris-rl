import warnings

import gymnasium as gym
import numpy as np

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*"
)

import pygame

import config
import tetris
from tetris import Tetris, Tetromino


class TetrisEnv(gym.Env):
    def __init__(self, render_mode = None):
        self.render_mode = render_mode

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(55,),
            dtype = np.float32
        )

        self.action_space = gym.spaces.Discrete(config.MAX_PLACEMENTS)
        
        if render_mode == "yes":
            pygame.init()
            self.screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
            pygame.display.set_caption('Tetris')
            self.clock=pygame.time.Clock()
        
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.game = Tetris(
            config.BOARD_WIDTH,
            config.BOARD_HEIGHT,
            self.np_random
        )

        return self.get_observation(), {}

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
                temp = Tetromino(x_pos, -2, piece.shape,self.game.rng,piece.shape_idx)
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

        for x in range(config.BOARD_WIDTH):
            found_filled = False
            for y in range(config.BOARD_HEIGHT):
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

    # excluding the rightmost col from bumpiness
    def compute_bumpiness(self):
        heights = []
        for x in range(config.BOARD_WIDTH - 1):  # cols 0..8
            height = 0
            for y in range(config.BOARD_HEIGHT):
                if self.game.grid[y][x] != 0:
                    height = config.BOARD_HEIGHT - y
                    break
            heights.append(height)
        return sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
    

    def column_height_std(self):
        heights = []
        for x in range(config.BOARD_WIDTH):
            height = 0
            for y in range(config.BOARD_HEIGHT):
                if self.game.grid[y][x] != 0:
                    height = config.BOARD_HEIGHT - y
                    break
            heights.append(height)
        mean = sum(heights) / len(heights)
        variance = sum((h - mean) ** 2 for h in heights) / len(heights)
        return variance ** 0.5

    def count_wells(self):
        heights = []
        for x in range(config.BOARD_WIDTH-1):
            height = 0
            for y in range(config.BOARD_HEIGHT):
                if self.game.grid[y][x] != 0:
                    height = config.BOARD_HEIGHT - y
                    break
            heights.append(height)

        wells = 0
        for i in range(len(heights)):
            left  = heights[i - 1] if i > 0 else config.BOARD_HEIGHT
            right = heights[i + 1] if i < len(heights) - 1 else config.BOARD_HEIGHT
            depth = min(left, right) - heights[i]
            if depth > 0:
                wells += depth
        return wells

    def right_well_depth(self):
        h8, h9 = 0, 0
        for y in range(config.BOARD_HEIGHT):
            if self.game.grid[y][8] != 0:
                h8 = config.BOARD_HEIGHT - y
                break
        for y in range(config.BOARD_HEIGHT):
            if self.game.grid[y][9] != 0:
                h9 = config.BOARD_HEIGHT - y
                break
        return max(0, h8 - h9)

    def near_complete_right_well_rows(self):
        count = 0
        for y in range(config.BOARD_HEIGHT):
            if all(self.game.grid[y][x] != 0 for x in range(9)) and self.game.grid[y][9] == 0:
                count += 1
        return count

    '''
    def compute_reward(self, lines_cleared, holes_before, holes_after,
                   height_after, bumpiness_after, game_over):
        line_rewards = [0, 5, 15, 50, 200]
        reward = line_rewards[lines_cleared]
#        if lines_cleared == 4:
#            reward += config.FAT_TET_BONUS

        reward += config.SURVIVAL_BONUS

        new_holes = holes_after - holes_before
        if new_holes > 0:
            reward -= config.NEW_HOLES_PENALTY * new_holes
        reward -= config.EXISTING_HOLES_PENALTY * holes_after
        reward -= config.HEIGHT_PENALTY * height_after
        reward -= config.BUMPINESS_PENALTY * bumpiness_after

        near = self.near_complete_right_well_rows()
        well = self.right_well_depth()
        reward += config.NEAR_COMPLETE_BONUS * near
        reward += config.RIGHT_WELL_BONUS * well

        if game_over:
            reward -= config.GAME_OVER_PENALTY

        return reward
    '''

    def compute_reward(self, lines_cleared, holes_before, holes_after,
                   height_after, bumpiness_after, game_over, phase):
        line_rewards = [0, 5, 15, 50, 200]
        reward = line_rewards[lines_cleared]

        reward += config.SURVIVAL_BONUS  

        new_holes = holes_after - holes_before
        if new_holes > 0:
            hole_scale = 1.0 + 4.0 * (phase ** 2)
            reward -= config.NEW_HOLES_PENALTY * hole_scale * new_holes

        reward -= config.HEIGHT_PENALTY * (1.0 + 3.0 * phase) * height_after
        reward -= config.BUMPINESS_PENALTY * (1.0 + 2.0 * phase) * bumpiness_after

        if game_over:
            reward -= config.GAME_OVER_PENALTY

        return reward

    def step(self, action):
        placements = self.get_valid_placement()
        action = action % len(placements)
        rotation, column = placements[action]

        holes_before = self.count_holes()

        self.game.current_piece.rotation = rotation
        self.game.current_piece.x = column
        self.game.current_piece.y = -2
        lines_cleared = self.game.hard_drop()

        holes_after = self.count_holes()
        height_after = self.aggregate_height()
        bumpiness_after = self.compute_bumpiness()
        phase = min(1.0, self.game.pieces / 80.0)

        reward = self.compute_reward(
            lines_cleared,
            holes_before,
            holes_after,
            height_after,
            bumpiness_after,
            self.game.game_over,
            phase
        )

        return self.get_observation(), reward, self.game.game_over, False, {}

    # to check what action indices are valid
    def action_masks(self):
        placements = self.get_valid_placement()
        num_valid = len(placements)

        array = np.zeros(config.MAX_PLACEMENTS, dtype=bool)

        for i in range(num_valid):
            array[i] = True

        return array

    # getting the states for the ppo
    def get_observation(self):
        obs = []
        for x in range(config.BOARD_WIDTH):
            height = 0
            for y in range(config.BOARD_HEIGHT):
                if self.game.grid[y][x] != 0:
                    height = config.BOARD_HEIGHT - y
                    break
            obs.append(height / config.BOARD_HEIGHT)

        for x in range(config.BOARD_WIDTH):
            col_holes = 0
            found_filled = False
            for y in range(config.BOARD_HEIGHT):
                if self.game.grid[y][x] != 0:
                    found_filled = True
                elif found_filled:
                    col_holes += 1
            obs.append(col_holes / config.BOARD_HEIGHT)

        obs.append(self.compute_bumpiness() / (config.BOARD_HEIGHT * 9))
        obs.append(self.aggregate_height() / (config.BOARD_HEIGHT * config.BOARD_WIDTH))
        obs.append(self.right_well_depth() / config.BOARD_HEIGHT)
        obs.append(self.near_complete_right_well_rows() / config.BOARD_HEIGHT)
        curr_arr = np.zeros(len(config.SHAPES))
        curr_arr[self.game.current_piece.shape_idx] = 1
        obs.extend(curr_arr)

        next_arr = np.zeros(len(config.SHAPES))
        next_arr[self.game.next_piece.shape_idx] = 1
        obs.extend(next_arr)

        for y in range(config.BOARD_HEIGHT):
            filled = sum(1 for x in range(config.BOARD_WIDTH) if self.game.grid[y][x] != 0)
            obs.append(filled / config.BOARD_WIDTH)

        phase = min(1.0, self.game.pieces / 80.0)
        obs.append(phase)

        return np.array(obs, dtype=np.float32)


    def render(self):
        if self.render_mode:
            self.screen.fill(config.BLACK)
            pygame.draw.rect(
                self.screen,
                config.WHITE,
                (
                    config.BOARD_X - 5,
                    config.BOARD_Y - 5,
                    config.PLAYFIELD_WIDTH + 10,
                    config.PLAYFIELD_HEIGHT + 10
                ),
                5
            )
            pygame.draw.rect(
                self.screen,
                config.WHITE,
                (config.NEXT_BOX_X - 5, config.NEXT_BOX_Y, 140, 120),
                4
            )
            tetris.draw_next_piece(self.screen, self.game.next_piece)

            tetris.draw(self.game,self.screen)
            if self.game.game_over:
                tetris.draw_game_over(self.screen,config.WIDTH // 2 - 100,config.HEIGHT // 2 - 30)

            tetris.draw_score(self.screen, self.game.score, self.game.lines, config.SCORE_BOX_X, config.SCORE_BOX_Y)

            pygame.display.flip()
            self.clock.tick(config.FPS)


