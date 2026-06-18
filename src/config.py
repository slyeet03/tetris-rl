# tetris wrapper constants
MAX_PLACEMENTS = 40
# penalties
SURVIVAL_BONUS = 3.0
NEW_HOLES_PENALTY = 2.5
GAME_OVER_PENALTY = 400
HEIGHT_PENALTY_DIFF = 1.5
BUMPINESS_PENALTY_DIFF = 1.0

EXISTING_HOLES_PENALTY = 0.002
HEIGHT_PENALTY = 0.005
BUMPINESS_PENALTY = 0.002
WELL_PENALTY = 0.01
SPREAD_PENALTY = 0.0
NEAR_COMPLETE_BONUS = 0.0
RIGHT_WELL_BONUS = 0.0
FAT_TET_BONUS    = 0
# DQN constants
DQN_NUM_STEPS = 8_000_000
DQN_BUFFER_SIZE = 750_000
DQN_BATCH_SIZE = 128
DQN_GAMMA = 0.995
DQN_LR = 1e-4
DQN_HIDDEN_SIZE = 256
DQN_TRAIN_FREQ = 4
DQN_TARGET_UPDATE_FREQ = 2000
DQN_WARMUP_STEPS = 10000
DQN_EPSILON_START = 1.0
DQN_EPSILON_END = 0.05
DQN_EPSILON_DECAY_STEPS = 3_000_000
DQN_BETA_START = 0.4
DQN_BETA_END = 1.0
DQN_BETA_ANNEAL_STEPS = 1_500_000
DQN_ALPHA = 0.6
DQN_CHECKPOINT_PATH = "../models/dqn_per_tetris.pt"
DQN_LOG_EVERY = 5000
# tetris constants
FPS = 60
FALL_SPEED = 60
GRID_SIZE = 30
PLAYFIELD_WIDTH = 300
PLAYFIELD_HEIGHT = 600
WIDTH = 550
HEIGHT = 650
BOARD_X = 50
BOARD_Y = 25
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
NEXT_BOX_X = 380
NEXT_BOX_Y = 25
SCORE_BOX_X = 380
SCORE_BOX_Y = 555
WHITE = (255, 255, 255)
BLACK = (0,0,0)
RED = (255,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
CYAN = (0,255,255)
YELLOW = (255,255,0)
MAGENTA = (255,0,255)
ORANGE = (255,165,0)
COLORS = [RED, BLUE, GREEN, CYAN, YELLOW, MAGENTA, ORANGE]
SHAPES = [
    [
        ['.....',
         '.....',
         '.....',
         '0000.',
         '.....'],
        ['.....',
         '..0..',
         '..0..',
         '..0..',
         '..0..']
    ],
    [
        ['.....',
         '.....',
         '..0..',
         '.000.',
         '.....'],
        ['.....',
         '..0..',
         '.00..',
         '..0..',
         '.....'],
        ['.....',
         '.....',
         '.000.',
         '..0..',
         '.....'],
        ['.....',
         '..0..',
         '..00.',
         '..0..',
         '.....']
    ],
    [
        [
         '.....',
         '.....',
         '..00.',
         '.00..',
         '.....'],
        ['.....',
         '.....',
         '.00..',
         '..00.',
         '.....'],
        ['.....',
         '.0...',
         '.00..',
         '..0..',
         '.....'],
        ['.....',
         '..0..',
         '.00..',
         '.0...',
         '.....']
    ],
    [
        ['.....',
         '..0..',
         '..0..',
         '..00.',
         '.....'],
        ['.....',
         '...0.',
         '.000.',
         '.....',
         '.....'],
        ['.....',
         '.00..',
         '..0..',
         '..0..',
         '.....'],
        ['.....',
         '.....',
         '.000.',
         '.0...',
         '.....']
    ],
    [
        ['.....',
         '.....',
         '..00.',
         '..00.',
         '.....'],
    ],
]
