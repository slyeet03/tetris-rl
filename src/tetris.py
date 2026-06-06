import random
import sys

import pygame

import config


class Tetromino:
    def __init__(self,x,y,shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = random.choice(config.COLORS)
        self.rotation = 0

class Tetris:
    def __init__(self,width,height):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.next_piece = self.new_piece()
        self.current_piece = self.new_piece()
        self.game_over = False
        self.score = 0
        self.lines = 0

    def new_piece(self):
        shape = random.choice(config.SHAPES)
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
            self.game_over = True

        return lines_cleared

    def update(self):
        if not self.game_over:
            if self.valid_move(self.current_piece, 0, 1, 0):
                self.current_piece.y += 1
            else:
                self.lock_piece(self.current_piece)

    def hard_drop(self):
        while self.valid_move(self.current_piece, 0, 1, 0):
            self.current_piece.y += 1
        
        lines_cleared = self.lock_piece(self.current_piece)

        return lines_cleared



def draw(game, screen):
        for y, row in enumerate(game.grid):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(
                        screen,
                        cell,
                        (
                            config.BOARD_X + x * config.GRID_SIZE,
                            config.BOARD_Y + y * config.GRID_SIZE,
                            config.GRID_SIZE - 1,
                            config.GRID_SIZE - 1
                        )
                    )
                    
        if game.current_piece:
            for i, row in enumerate(game.current_piece.shape[game.current_piece.rotation % len(game.current_piece.shape)]):
                for j, cell in enumerate(row):
                    if cell == '0':
                        pygame.draw.rect(
                            screen,
                            game.current_piece.color,
                            (
                                config.BOARD_X + (game.current_piece.x + j) * config.GRID_SIZE,
                                config.BOARD_Y + (game.current_piece.y + i) * config.GRID_SIZE,
                                config.GRID_SIZE - 1,
                                config.GRID_SIZE - 1
                            )
                        )


def draw_score(screen, score, lines, x, y):
    font = pygame.font.Font(None, 30)
    box_w = 130
    row_h = 28

    pygame.draw.rect(screen, config.WHITE, (x, y, box_w, row_h * 2), 2)
    pygame.draw.line(screen, config.WHITE, (x, y + row_h), (x + box_w, y + row_h), 2)

    mid = x + box_w // 2
    pygame.draw.line(screen, config.WHITE, (mid, y), (mid, y + row_h * 2), 2)

    label_font = pygame.font.Font(None, 26)
    screen.blit(label_font.render("Score", True, config.WHITE), (x + 6, y + 7))
    screen.blit(label_font.render("Lines", True, config.WHITE), (x + 6, y + row_h + 7))
    screen.blit(label_font.render(str(score), True, config.WHITE), (mid + 6, y + 7))
    screen.blit(label_font.render(str(lines), True, WHITE), (mid + 6, y + row_h + 7))

def draw_next_piece(screen, piece):
    font = pygame.font.Font(None, 36)
    text = font.render("Next", True, config.WHITE)
    screen.blit(text, (config.NEXT_BOX_X + 10, config.NEXT_BOX_Y + 8))

    shape = piece.shape[0]
    cells = [(i, j) for i, row in enumerate(shape) for j, cell in enumerate(row) if cell == '0']
    if not cells:
        return

    min_i = min(c[0] for c in cells)
    max_i = max(c[0] for c in cells)
    min_j = min(c[1] for c in cells)
    max_j = max(c[1] for c in cells)

    piece_w = (max_j - min_j + 1) * config.GRID_SIZE
    piece_h = (max_i - min_i + 1) * config.GRID_SIZE

    box_content_y = config.NEXT_BOX_Y + 35
    box_content_h = 85

    start_x = config.NEXT_BOX_X + (140 - piece_w) // 2
    start_y = box_content_y + (box_content_h - piece_h) // 2

    for i, row in enumerate(shape):
        for j, cell in enumerate(row):
            if cell == '0':
                pygame.draw.rect(
                    screen,
                    piece.color,
                    (
                        start_x + (j - min_j) * config.GRID_SIZE,
                        start_y + (i - min_i) * config.GRID_SIZE,
                        config.GRID_SIZE - 1,
                        config.GRID_SIZE - 1
                    )
                )

def draw_game_over(screen, x, y):
    font = pygame.font.Font(None, 48)
    text = font.render("Game Over", True, config.RED)
    screen.blit(text, (x, y))

def tetris():
    pygame.init()

    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
    pygame.display.set_caption('Tetris')
    clock=pygame.time.Clock()

    game = Tetris(config.BOARD_WIDTH,config.BOARD_HEIGHT)

    fall_time = 0
    fall_speed = config.FALL_SPEED

    while True:
        screen.fill(config.BLACK)
        pygame.draw.rect(
            screen,
            config.WHITE,
            (
                config.BOARD_X - 5,
                config.BOARD_Y - 5,
                config.PLAYFIELD_WIDTH + 10,
                config.PLAYFIELD_HEIGHT + 10
            ),
            5
        )
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if game.game_over:
                    game = Tetris(config.BOARD_WIDTH,config.BOARD_HEIGHT)
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
            config.WHITE,
            (config.NEXT_BOX_X - 5, config.NEXT_BOX_Y, 140, 120),
            4
        )
        draw_next_piece(screen, game.next_piece)

        draw(game,screen)
        if game.game_over:
            draw_game_over(screen,config.WIDTH // 2 - 100,config.HEIGHT // 2 - 30)

        draw_score(screen, game.score, game.lines, config.SCORE_BOX_X, config.SCORE_BOX_Y)

        pygame.display.flip()
        clock.tick(config.FPS)

if __name__ == "__main__":
    tetris()
