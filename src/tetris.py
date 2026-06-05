import random
import sys

import pygame

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
]

class Tetromino:
    def __init__(self,x,y,shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = random.choice(COLORS)
        self.rotation = 0

class Tetris:
    def __init__(self,width,height):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.next_piece = self.new_piece()
        self.current_piece = self.new_piece()
        self.gamer_over = False
        self.score = 0
        self.lines = 0

    def new_piece(self):
        shape = random.choice(SHAPES)
        return Tetromino(3, -2, shape)

    def valid_move(self, piece, x, y, rotation):
        for i, row in enumerate(piece.shape[(piece.rotation+rotation) % len(piece.shape)]):
            for j, cell in enumerate(row):
                if cell != '0':
                    continue
                try:
                    new_x = piece.x + j + x
                    new_y = piece.y + i + y

                    if new_x < 0 or new_x >= self.width:
                        return False

                    if new_y < 0:
                        continue

                    if new_y >= self.height:
                        return False

                    if self.grid[new_y][new_x] != 0:
                        return False
                except IndexError:
                    return False
        return True

    def clear_lines(self):
        lines_cleared = 0
        i = self.height - 1
        while i >= 0:
            if all(cell != 0 for cell in self.grid[i]):
                lines_cleared += 1
                del self.grid[i]
                self.grid.insert(0, [0 for _ in range(self.width)])
            else:
                i -= 1
        return lines_cleared

    def lock_piece(self,piece):
        for i, row in enumerate(piece.shape[piece.rotation % len(piece.shape)]):
            for j, cell in enumerate(row):
                if cell == '0':
                    self.grid[piece.y + i][piece.x + j] = piece.color

        lines_cleared = self.clear_lines()
        self.score += lines_cleared * 100
        self.lines += lines_cleared
        self.current_piece = self.next_piece
        self.next_piece = self.new_piece()

        if not self.valid_move(self.current_piece,0,0,0):
            self.gamer_over = True

        return lines_cleared

    def update(self):
        if not self.gamer_over:
            if self.valid_move(self.current_piece, 0, 1, 0):
                self.current_piece.y += 1
            else:
                self.lock_piece(self.current_piece)

    def draw(self, screen):
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(
                        screen,
                        cell,
                        (
                            BOARD_X + x * GRID_SIZE,
                            BOARD_Y + y * GRID_SIZE,
                            GRID_SIZE - 1,
                            GRID_SIZE - 1
                        )
                    )
                    
        if self.current_piece:
            for i, row in enumerate(self.current_piece.shape[self.current_piece.rotation % len(self.current_piece.shape)]):
                for j, cell in enumerate(row):
                    if cell == '0':
                        pygame.draw.rect(
                            screen,
                            self.current_piece.color,
                            (
                                BOARD_X + (self.current_piece.x + j) * GRID_SIZE,
                                BOARD_Y + (self.current_piece.y + i) * GRID_SIZE,
                                GRID_SIZE - 1,
                                GRID_SIZE - 1
                            )
                        )

def draw_score(screen, score, lines, x, y):
    font = pygame.font.Font(None, 30)
    box_w = 130
    row_h = 28

    pygame.draw.rect(screen, WHITE, (x, y, box_w, row_h * 2), 2)
    pygame.draw.line(screen, WHITE, (x, y + row_h), (x + box_w, y + row_h), 2)

    mid = x + box_w // 2
    pygame.draw.line(screen, WHITE, (mid, y), (mid, y + row_h * 2), 2)

    label_font = pygame.font.Font(None, 26)
    screen.blit(label_font.render("Score", True, WHITE), (x + 6, y + 7))
    screen.blit(label_font.render("Lines", True, WHITE), (x + 6, y + row_h + 7))
    screen.blit(label_font.render(str(score), True, WHITE), (mid + 6, y + 7))
    screen.blit(label_font.render(str(lines), True, WHITE), (mid + 6, y + row_h + 7))

def draw_next_piece(screen, piece):
    font = pygame.font.Font(None, 36)
    text = font.render("Next", True, WHITE)
    screen.blit(text, (NEXT_BOX_X + 10, NEXT_BOX_Y + 8))

    shape = piece.shape[0]
    cells = [(i, j) for i, row in enumerate(shape) for j, cell in enumerate(row) if cell == '0']
    if not cells:
        return

    min_i = min(c[0] for c in cells)
    max_i = max(c[0] for c in cells)
    min_j = min(c[1] for c in cells)
    max_j = max(c[1] for c in cells)

    piece_w = (max_j - min_j + 1) * GRID_SIZE
    piece_h = (max_i - min_i + 1) * GRID_SIZE

    box_content_y = NEXT_BOX_Y + 35
    box_content_h = 85

    start_x = NEXT_BOX_X + (140 - piece_w) // 2
    start_y = box_content_y + (box_content_h - piece_h) // 2

    for i, row in enumerate(shape):
        for j, cell in enumerate(row):
            if cell == '0':
                pygame.draw.rect(
                    screen,
                    piece.color,
                    (
                        start_x + (j - min_j) * GRID_SIZE,
                        start_y + (i - min_i) * GRID_SIZE,
                        GRID_SIZE - 1,
                        GRID_SIZE - 1
                    )
                )

def draw_game_over(screen, x, y):
    font = pygame.font.Font(None, 48)
    text = font.render("Game Over", True, RED)
    screen.blit(text, (x, y))

def tetris():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Tetris')
    clock=pygame.time.Clock()

    game = Tetris(BOARD_WIDTH,BOARD_HEIGHT)

    fall_time = 0
    fall_speed = FALL_SPEED

    while True:
        screen.fill(BLACK)
        pygame.draw.rect(
            screen,
            WHITE,
            (
                BOARD_X - 5,
                BOARD_Y - 5,
                PLAYFIELD_WIDTH + 10,
                PLAYFIELD_HEIGHT + 10
            ),
            5
        )
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if game.gamer_over:
                    game = Tetris(BOARD_WIDTH,BOARD_HEIGHT)
                else:
                    if event.key == pygame.K_LEFT:
                        if game.valid_move(game.current_piece, -1, 0, 0):
                            game.current_piece.x -= 1

                    if event.key == pygame.K_RIGHT:
                        if game.valid_move(game.current_piece, 1, 0, 0):
                            game.current_piece.x += 1

                    if event.key == pygame.K_DOWN:
                        if game.valid_move(game.current_piece, 0, 1, 0):
                            game.current_piece.y += 1

                    if event.key == pygame.K_UP:
                        if game.valid_move(game.current_piece, 0, 0, 1):
                            game.current_piece.rotation += 1

                    if event.key == pygame.K_SPACE:
                        while game.valid_move(game.current_piece, 0, 1, 0):
                            game.current_piece.y += 1
                        game.lock_piece(game.current_piece)

        delta_time = clock.get_rawtime()
        fall_time+=delta_time
        if fall_time >= fall_speed:
            game.update()
            fall_time=0

        pygame.draw.rect(
            screen,
            WHITE,
            (NEXT_BOX_X - 5, NEXT_BOX_Y, 140, 120),
            4
        )
        draw_next_piece(screen, game.next_piece)

        game.draw(screen)
        if game.gamer_over:
            draw_game_over(screen,WIDTH // 2 - 100,HEIGHT // 2 - 30)

        draw_score(screen, game.score, game.lines, SCORE_BOX_X, SCORE_BOX_Y)

        pygame.display.flip()
        clock.tick(FPS)

tetris()
