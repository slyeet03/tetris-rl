# tetris wrapper constants
MAX_PLACEMENTS = 40
# penalties
SURVIVAL_BONUS = 2.0
NEW_HOLES_PENALTY = 2.5
EXISTING_HOLES_PENALTY = 0.01
HEIGHT_PENALTY = 0.02
BUMPINESS_PENALTY = 0.04
GAME_OVER_PENALTY = 30

HEIGHT_PENALTY_DIFF = 0.0
BUMPINESS_PENALTY_DIFF = 0.0
WELL_PENALTY = 0.0
SPREAD_PENALTY = 0.0
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
