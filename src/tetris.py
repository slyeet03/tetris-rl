import random
import sys

import pygame

FPS = 60
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 25
WHITE = (255, 255, 255)
BLACK = (0,0,0)
RED = (255,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
COLORS = [RED,BLUE,GREEN]
SHAPES = [
    # I
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
    # T
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
    # Z
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
    # L
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
        self.current_piece = self.new_piece()
        self.gamer_over = False
        self.score = 0

    def new_piece(self):
        shape = random.choice(SHAPES)
        return Tetromino(self.width // 2, 0, shape)

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
        self.score+= lines_cleared*100
        self.current_piece = self.new_piece()

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
                    pygame.draw.rect(screen, cell,(x*GRID_SIZE, y*GRID_SIZE, GRID_SIZE-1,GRID_SIZE-1))

        if self.current_piece:
            for i, row in enumerate(self.current_piece.shape[self.current_piece.rotation % len(self.current_piece.shape)]):
                for j, cell in enumerate(row):
                    if cell == '0':
                        pygame.draw.rect(screen, self.current_piece.color, ((self.current_piece.x + j) * GRID_SIZE, (self.current_piece.y + i) * GRID_SIZE, GRID_SIZE - 1, GRID_SIZE - 1))

def draw_score(screen, score, x, y):
    font = pygame.font.Font(None, 36)
    text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(text, (x, y))


def draw_game_over(screen, x, y):
    font = pygame.font.Font(None, 48)
    text = font.render("Game Over", True, RED)
    screen.blit(text, (x, y))

def tetris():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Tetris')
    clock=pygame.time.Clock()

    game = Tetris(WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE)

    fall_time = 0
    fall_speed = 50

    while True:
        screen.fill(BLACK)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if game.gamer_over:
                    game = Tetris(WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE)
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

        draw_score(screen,game.score,10,10)

        game.draw(screen)
        if game.gamer_over:
            draw_game_over(screen,WIDTH // 2 - 100,HEIGHT // 2 - 30)

        pygame.display.flip()
        clock.tick(FPS)

tetris()
