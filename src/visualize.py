# this file was made using AI -> i suck at visualization
import os
import time
import warnings

import numpy as np
import pygame
import torch

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*"
)
import config
import tetris as tet
from DQN import ValueNetwork
from tetris_wrapper import TetrisEnv

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "models", "dqn_per_tetris.pt")
GAME_W   = config.WIDTH
VIZ_W    = 440
TOTAL_W  = GAME_W + VIZ_W
TOTAL_H  = config.HEIGHT

STEP_DELAY = 0.5

BG_PANEL    = (18, 18, 28)
COL_ACCENT  = (0, 200, 255)
COL_GOOD    = (55, 215, 85)
COL_BAD     = (215, 65, 65)
COL_WARN    = (220, 175, 40)
COL_TEXT    = (210, 210, 210)
COL_DIM     = (105, 105, 128)
COL_CHOSEN  = (50, 255, 50)

N_COLS  = config.BOARD_WIDTH
N_SHAPES = len(config.SHAPES)

# layout of build_afterstate_features() in tetris_wrapper.py:
#   [0:N_COLS)                 heights
#   [N_COLS:2*N_COLS)          holes per column
#   2*N_COLS                   bumpiness
#   2*N_COLS+1                 aggregate height
#   2*N_COLS+2                 wells
#   [2*N_COLS+3 : 2*N_COLS+8)  lines-cleared one-hot (5)
#   next N_SHAPES               current/next piece one-hot
#   next N_SHAPES               next-next piece one-hot
_HEIGHTS_SLICE   = slice(0, N_COLS)
_HOLES_SLICE     = slice(N_COLS, 2 * N_COLS)
_BUMPINESS_IDX   = 2 * N_COLS
_AGG_HEIGHT_IDX  = 2 * N_COLS + 1
_WELLS_IDX       = 2 * N_COLS + 2
_LINES_SLICE     = slice(2 * N_COLS + 3, 2 * N_COLS + 8)
_PIECE_SLICE     = slice(2 * N_COLS + 8, 2 * N_COLS + 8 + N_SHAPES)
_NEXT_PIECE_SLICE = slice(2 * N_COLS + 8 + N_SHAPES, 2 * N_COLS + 8 + 2 * N_SHAPES)


def lerp_color(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def get_value_info(v_net, candidates, gamma, device):
    """Scores every candidate afterstate the same way DQNAgent.select_candidate
       does, and returns extra info for the visualization:
         - probs: softmax over the candidates' scores, used ONLY for the
           heatmap / bar viz below. DQN never samples from this -- it always
           argmaxes -- this just shows how much better the chosen placement
           looked relative to the others.
         - scores: reward + gamma * V(afterstate) * (1 - done) for every
           candidate. This is exactly what selection ranks on.
         - values: raw V(afterstate) from the network, kept for reference.
    """
    feats = np.stack([c["features"] for c in candidates]).astype(np.float32)
    rewards = np.array([c["reward"] for c in candidates], dtype=np.float32)
    dones = np.array([c["done"] for c in candidates], dtype=np.float32)

    with torch.no_grad():
        feats_t = torch.as_tensor(feats, dtype=torch.float32, device=device)
        values = v_net(feats_t).cpu().numpy()

    scores = rewards + gamma * values * (1.0 - dones)

    exp_s = np.exp(scores - scores.max())
    probs = exp_s / exp_s.sum()

    return probs, scores, values


def get_ghost_cells(game, rotation, x_pos):
    from tetris import Tetromino
    piece = game.current_piece
    temp  = Tetromino(x_pos, -2, piece.shape, game.rng, piece.shape_idx)
    temp.rotation = rotation

    while game.valid_move(temp, 0, 1, 0):
        temp.y += 1

    cells = []
    for i, row in enumerate(temp.shape[temp.rotation % len(temp.shape)]):
        for j, cell in enumerate(row):
            if cell == '0':
                bx, by = temp.x + j, temp.y + i
                if 0 <= bx < game.width and 0 <= by < game.height:
                    cells.append((bx, by))
    return cells


def draw_placement_heatmap(screen, game, placements, probs, chosen_idx):
    if not placements:
        return

    max_p = max(probs[:len(placements)].max(), 1e-9)
    cell  = pygame.Surface((config.GRID_SIZE - 1, config.GRID_SIZE - 1), pygame.SRCALPHA)

    # draw non-chosen placements first so chosen renders on top
    order = [i for i in range(len(placements)) if i != chosen_idx] + [chosen_idx]

    for idx in order:
        rot, col = placements[idx]
        norm_p   = probs[idx] / max_p

        if idx == chosen_idx:
            rgba = (50, 255, 50, 210)
        else:
            r, g, b = lerp_color((20, 30, 140), (255, 50, 20), norm_p)
            rgba    = (r, g, b, max(15, int(155 * norm_p)))

        cell.fill(rgba)
        for bx, by in get_ghost_cells(game, rot, col):
            screen.blit(
                cell,
                (config.BOARD_X + bx * config.GRID_SIZE,
                 config.BOARD_Y + by * config.GRID_SIZE),
            )


def draw_section_title(screen, text, font, x, y):
    surf = font.render(text, True, COL_ACCENT)
    screen.blit(surf, (x, y))
    pygame.draw.line(screen, COL_ACCENT, (x, y + 17), (x + surf.get_width(), y + 17), 1)
    return y + 22


def draw_viz_panel(screen, chosen_features, placements, probs, chosen_idx, state_val,
                   font_lg, font_md, font_sm, px):

    pygame.draw.rect(screen, BG_PANEL, (GAME_W, 0, VIZ_W, TOTAL_H))
    W = VIZ_W - 20   # usable content width
    cy = 10

    # Value estimate: V(afterstate) for the chosen placement
    t    = min(1.0, max(0.0, (state_val + 150) / 650))
    vcol = lerp_color(COL_BAD, COL_GOOD, t)
    screen.blit(font_lg.render(f"V  {state_val:+.1f}", True, vcol), (px, cy))
    cy += 38
    screen.blit(font_sm.render("V(afterstate) for chosen placement  (red = bad board, green = good)", True, COL_DIM), (px, cy))
    cy += 20

    # Entropy of the softmax-over-score visualization (NOT a real policy
    # distribution -- selection always argmaxes -- this is just "how close
    # was the runner-up placement's score to the winner's".
    p_valid = probs[:len(placements)]
    p_valid = p_valid[p_valid > 1e-9]
    entropy = float(-np.sum(p_valid * np.log(p_valid)))
    ecol    = lerp_color(COL_GOOD, COL_BAD, min(1.0, entropy / 3.0))
    screen.blit(font_md.render(f"H  {entropy:.2f} nats", True, ecol), (px, cy))
    cy += 26
    screen.blit(font_sm.render("score spread  (↓ clear winner  ↑ close call)", True, COL_DIM), (px, cy))
    cy += 20

    pygame.draw.line(screen, COL_DIM, (px, cy), (px + W, cy))
    cy += 10

    # Top placements
    cy = draw_section_title(screen, "TOP PLACEMENTS", font_sm, px, cy)
    n      = len(placements)
    ranked = sorted(range(n), key=lambda i: probs[i], reverse=True)[:6]
    best_p = max(probs[:n].max(), 1e-9)

    for idx in ranked:
        rot, col = placements[idx]
        prob     = probs[idx]
        is_chosen = (idx == chosen_idx)
        c = COL_CHOSEN if is_chosen else COL_TEXT

        marker = "▶" if is_chosen else " "
        screen.blit(font_sm.render(f"{marker} rot={rot}  x={col:+d}", True, c), (px, cy))
        pct_surf = font_sm.render(f"{prob * 100:.1f}%", True, c)
        screen.blit(pct_surf, (px + W - pct_surf.get_width(), cy))

        # proportion bar
        bar_len = int((prob / best_p) * (W - 70))
        bar_col = COL_CHOSEN if is_chosen else lerp_color(COL_DIM, COL_ACCENT, prob / best_p)
        pygame.draw.rect(screen, bar_col, (px, cy + 14, bar_len, 4))
        cy += 26

    pygame.draw.line(screen, COL_DIM, (px, cy), (px + W, cy))
    cy += 10

    # Chosen-afterstate feature bars
    cy = draw_section_title(screen, "CHOSEN AFTERSTATE FEATURES", font_sm, px, cy)

    obs = chosen_features

    screen.blit(
        font_sm.render(f"Bumpiness: {obs[_BUMPINESS_IDX]:.3f}", True, COL_TEXT),
        (px, cy)
    )
    cy += 18

    screen.blit(
        font_sm.render(f"Aggregate Height: {obs[_AGG_HEIGHT_IDX]:.3f}", True, COL_TEXT),
        (px, cy)
    )
    cy += 18

    screen.blit(
        font_sm.render(f"Wells: {obs[_WELLS_IDX]:.3f}", True, COL_TEXT),
        (px, cy)
    )
    cy += 24

    groups = [
        (f"Column heights [0-{N_COLS-1}]", obs[_HEIGHTS_SLICE], COL_WARN),
        (f"Column holes [{N_COLS}-{2*N_COLS-1}]", obs[_HOLES_SLICE], COL_BAD),
        ("Current piece one-hot", obs[_PIECE_SLICE], (190, 95, 255)),
        ("Next piece one-hot", obs[_NEXT_PIECE_SLICE], (255, 145, 45)),
    ]

    remaining_h = TOTAL_H - cy - 30
    row_h = max(32, remaining_h // len(groups))
    BAR_MAX_H = row_h - 16

    for label, vals, colour in groups:
        screen.blit(font_sm.render(label, True, COL_DIM), (px, cy))
        cy += 13

        n_bars = len(vals)

        if n_bars == 0:
            cy += BAR_MAX_H + 4
            continue

        bw = max(1, (W - n_bars + 1) // n_bars)

        for i, v in enumerate(vals):
            bh = max(1, int(v * BAR_MAX_H))

            bx = px + i * (bw + 1)
            by = cy + BAR_MAX_H - bh

            pygame.draw.rect(
                screen,
                colour,
                (bx, by, bw, bh)
            )

            if label.startswith("Column heights") and v > 0.05:
                pygame.draw.line(
                    screen,
                    config.WHITE,
                    (bx, by),
                    (bx + bw, by),
                    1
                )

        cy += BAR_MAX_H + 4

    # Footer
    screen.blit(
        font_sm.render(f"↑/↓ speed  |  delay={STEP_DELAY:.2f}s", True, COL_DIM),
        (px, TOTAL_H - 18),
    )


def main():
    global STEP_DELAY

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Loading model from {MODEL_PATH}")
    env = TetrisEnv(render_mode=None)   # we own the pygame window

    v_net = ValueNetwork(config.AFTERSTATE_DIM).to(device)
    v_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    v_net.eval()

    gamma = config.DQN_GAMMA

    pygame.init()
    screen  = pygame.display.set_mode((TOTAL_W, TOTAL_H))
    clk     = pygame.time.Clock()
    font_lg = pygame.font.Font(None, 50)
    font_md = pygame.font.Font(None, 32)
    font_sm = pygame.font.Font(None, 19)
    PX      = GAME_W + 10  # panel x start

    game_number = 0
    running     = True

    while running:
        game_number += 1
        candidates = env.reset()
        done       = False
        pygame.display.set_caption(f"Tetris AI Visualizer — Game {game_number}")

        while not done and running and candidates:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        STEP_DELAY = max(0.05, STEP_DELAY - 0.05)
                        print(f"delay → {STEP_DELAY:.2f}s")
                    if event.key == pygame.K_DOWN:
                        STEP_DELAY = min(2.0, STEP_DELAY + 0.05)
                        print(f"delay → {STEP_DELAY:.2f}s")

            # ── Value-network inference over every candidate afterstate ────
            probs, scores, values = get_value_info(v_net, candidates, gamma, device)
            chosen_idx = int(np.argmax(scores))
            chosen     = candidates[chosen_idx]
            placements = [(c["rotation"], c["x_pos"]) for c in candidates]

            # ── Render game board ───────────────────────────────────────────
            screen.fill(config.BLACK)

            pygame.draw.rect(screen, config.WHITE,
                (config.BOARD_X - 5, config.BOARD_Y - 5,
                 config.PLAYFIELD_WIDTH + 10, config.PLAYFIELD_HEIGHT + 10), 5)
            pygame.draw.rect(screen, config.WHITE,
                (config.NEXT_BOX_X - 5, config.NEXT_BOX_Y, 140, 120), 4)

            tet.draw_next_piece(screen, env.game.next_piece)
            tet.draw(env.game, screen)
            tet.draw_score(screen, env.game.score, env.game.lines,
                           config.SCORE_BOX_X, config.SCORE_BOX_Y)

            if env.game.game_over:
                tet.draw_game_over(screen,
                    config.WIDTH // 2 - 100, config.HEIGHT // 2 - 30)

            # placement probability heatmap on top of the board
            draw_placement_heatmap(screen, env.game, placements, probs, chosen_idx)

            # ── Render neural network panel ───────────────────────────────────
            draw_viz_panel(screen, chosen["features"], placements, probs, chosen_idx,
                           values[chosen_idx], font_lg, font_md, font_sm, PX)

            pygame.display.flip()
            clk.tick(60)
            time.sleep(STEP_DELAY)

            # ── Apply the chosen placement (this is the actual env step) ───
            env.apply_placement(chosen["rotation"], chosen["x_pos"])
            done = chosen["done"]
            candidates = [] if done else env.get_candidates()

        if done:
            print(f"Game {game_number:3d} — Score: {env.game.score:5d} | Lines: {env.game.lines:3d}")
            time.sleep(1.0)

    pygame.quit()
    print(f"Stopped after {game_number} games.")


if __name__ == "__main__":
    main()
