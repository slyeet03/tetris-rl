import warnings

import numpy as np

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*"
)

import pygame

import config
import tetris
from tetris import Tetris, Tetromino


class TetrisEnv:
    def __init__(self, render_mode = None, seed=None):
        self.render_mode = render_mode
        self.rng=np.random.default_rng(seed)
        self.game=None

        if render_mode == "yes":
            pygame.init()
            self.screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
            pygame.display.set_caption('Tetris')
            self.clock=pygame.time.Clock()
        
        self.reset()

    def reset(self, *, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.game = Tetris(
            config.BOARD_WIDTH,
            config.BOARD_HEIGHT,
            self.rng
        )

        return self.get_candidates()

    def count_holes(self, grid=None):
        if grid is None:
            grid = self.game.grid

        total_holes = 0

        for x in range(config.BOARD_WIDTH):
            found_filled = False
            for y in range(config.BOARD_HEIGHT):
                if self.game.grid[y][x] != 0:
                    found_filled = True
                elif found_filled:
                    total_holes += 1

        return total_holes

    def aggregate_height(self, grid=None):
        if grid is None:
            grid = self.game.grid

        total = 0

        for x in range(config.BOARD_WIDTH):
            for y in range(config.BOARD_HEIGHT):
                if self.game.grid[y][x] != 0:
                    total += (config.BOARD_HEIGHT - y)
                    break

        return total

    # excluding the rightmost col from bumpiness
    def compute_bumpiness(self, grid=None):
        if grid is None:
            grid = self.game.grid

        heights = []
        for x in range(config.BOARD_WIDTH - 1):  # cols 0..8
            height = 0
            for y in range(config.BOARD_HEIGHT):
                if self.game.grid[y][x] != 0:
                    height = config.BOARD_HEIGHT - y
                    break
            heights.append(height)
        return sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
    

    def count_wells(self, grid=None):
        if grid is None:
            grid = self.game.grid

        heights = []
        for x in range(config.BOARD_WIDTH):
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

    
    

    
    def compute_reward(self,lines_cleared, holes_before,holes_after,height_before,height_after,bumpiness_before,bumpiness_after,game_over,grid_after=None):
        reward = 0

        line_rewards = [0,10,80,150,500]

        new_holes = holes_after - holes_before
        height_diff = height_after - height_before
        bumpiness_diff = bumpiness_after - bumpiness_before

        reward += line_rewards[lines_cleared]
        reward += config.SURVIVAL_BONUS

        if new_holes > 0:
            reward -= config.NEW_HOLES_PENALTY * new_holes
        elif new_holes < 0:
            reward += 0.5 * config.NEW_HOLES_PENALTY * abs(new_holes)

        if height_diff > 0:
            reward -= config.HEIGHT_PENALTY_DIFF * height_diff
        elif height_diff < 0:
            reward += 0.5 * config.HEIGHT_PENALTY_DIFF * abs(height_diff)

        if bumpiness_diff > 0:
            reward -= config.BUMPINESS_PENALTY_DIFF * bumpiness_diff
        elif bumpiness_diff < 0:
            reward += 0.5 * config.BUMPINESS_PENALTY_DIFF * abs(bumpiness_diff)
 
        reward -= config.WELL_PENALTY * self.count_wells()

        if game_over:
            reward -= config.GAME_OVER_PENALTY

        return reward

    def simulate_drop(self, shape, x_pos, y_pos):
        grid_copy = [row.copy() for row in self.game.grid]
        
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell == '0':
                    y=y_pos+i
                    x=x_pos+j
                    if 0<= y < config.BOARD_HEIGHT:
                        grid_copy[y][x] = 1

        lines_cleared=0
        i=config.BOARD_HEIGHT-1
        while i>= 0:
            if all(c != 0 for c in grid_copy[i]):
                lines_cleared+=1
                del grid_copy[i]
                grid_copy.insert(0,[0 for _ in range(config.BOARD_WIDTH)])
            else:
                i -= 1
        return grid_copy, lines_cleared

    # whether putting that piece there is valid or not
    def check_spawn_blocked(self, grid, piece):
        shape = piece.shape[0]
        spawn_x, spawn_y = 3, -2
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell != '0':
                    continue

                x=spawn_x + j
                y=spawn_y+i

                # wall collision
                if x<0 or x>=config.BOARD_WIDTH:
                    return True
                # bottom of the earth
                if y<0:
                    continue
                # out of bound
                if y>=config.BOARD_HEIGHT:
                    return True
                # colliding with another block
                if grid[y][x] != 0:
                    return True
        
        return False

    # input for the network
    def build_afterstate_features(self, grid, lines_cleared, next_piece_idx, next_next_piece_idx):
        heights=[]
        holes_per_col=[]
        for x in range(config.BOARD_WIDTH):
            height=0
            found_filled = False
            col_holes = 0
           
            for y in range(config.BOARD_HEIGHT):
                if grid[y][x] != 0:
                    if not found_filled:
                        height = config.BOARD_HEIGHT - y

                    found_filled = True
           
                elif found_filled:
                    col_holes += 1

            heights.append(height / config.BOARD_HEIGHT)
            holes_per_col.append(col_holes / config.BOARD_HEIGHT)

        bumpiness = self.compute_bumpiness(grid) / (config.BOARD_HEIGHT * (config.BOARD_WIDTH - 1))
        agg_height = self.aggregate_height(grid) / (config.BOARD_HEIGHT * config.BOARD_WIDTH)
        wells = self.count_wells(grid) / (config.BOARD_HEIGHT * config.BOARD_WIDTH)

        lines_onehot = np.zeros(5, dtype=np.float32)
        lines_onehot[min(lines_cleared, 4)] = 1.0

        piece_onehot = np.zeros(len(config.SHAPES), dtype=np.float32)
        piece_onehot[next_piece_idx] = 1.0

        next_piece_onehot = np.zeros(len(config.SHAPES), dtype=np.float32)
        next_piece_onehot[next_next_piece_idx] = 1.0

        return np.concatenate([
            np.array(heights, dtype=np.float32),
            np.array(holes_per_col, dtype=np.float32),
            np.array([bumpiness, agg_height, wells], dtype=np.float32),
            lines_onehot,
            piece_onehot,
            next_piece_onehot,
        ])

    def get_candidates(self):
        piece = self.game.current_piece
        num_rotations = len(piece.shape)
 
        holes_before = self.count_holes()
        height_before = self.aggregate_height()
        bumpiness_before = self.compute_bumpiness()
 
        candidates = []
 
        for rotation_idx in range(num_rotations):
            shape = piece.shape[rotation_idx]
 
            occupied = [
                (i, j)
                for i, row in enumerate(shape)
                for j, cell in enumerate(row)
                if cell == '0'
            ]
            leftmost = min(j for _, j in occupied)
            rightmost = max(j for _, j in occupied)
 
            for x_pos in range(-leftmost, self.game.width - rightmost):
                temp = Tetromino(x_pos, -2, piece.shape, self.game.rng, piece.shape_idx)
                temp.rotation = rotation_idx
 
                if not self.game.valid_move(temp, 0, 0, 0):
                    continue
 
                # hard drop
                while self.game.valid_move(temp, 0, 1, 0):
                    temp.y += 1
 
                grid_after, lines_cleared = self.simulate_drop(shape, x_pos, temp.y)
 
                holes_after = self.count_holes(grid_after)
                height_after = self.aggregate_height(grid_after)
                bumpiness_after = self.compute_bumpiness(grid_after)
 
                game_over = self.check_spawn_blocked(grid_after, self.game.next_piece)
 
                reward = self.compute_reward(
                    lines_cleared, holes_before, holes_after,
                    height_before, height_after,
                    bumpiness_before, bumpiness_after,
                    game_over, grid_after
                )
 
                features = self.build_afterstate_features(
                    grid_after, lines_cleared,
                    self.game.next_piece.shape_idx,
                    self.game.next_next_piece.shape_idx
                )
 
                candidates.append({
                    "rotation": rotation_idx,
                    "x_pos": x_pos,
                    "features": features,
                    "reward": reward,
                    "done": game_over,
                })
 
        return candidates

    def apply_placement(self, rotation, x_pos):
        self.game.current_piece.rotation = rotation
        self.game.current_piece.x=x_pos
        self.game.current_piece.y=-2

        return self.game.hard_drop()

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


