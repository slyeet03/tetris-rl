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
from DQN import QNetwork
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


def lerp_color(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def get_q_info(q_net, obs, action_mask, device):
    """Runs the Q-network on obs and returns:
       - probs: a softmax over the masked Q-values, used ONLY for the
         heatmap / bar visualization below. DQN never actually samples
         from this -- it always argmaxes -- this just shows how much
         better the chosen placement looked relative to the others.
       - value: max masked Q-value, the closest DQN analog to a critic's
         state-value estimate (there's no separate value head here).
       - q_values: the raw, unmasked network output, kept for reference.
    """
    obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        q_values = q_net(obs_t).squeeze(0).cpu().numpy()

    masked_q = np.where(action_mask, q_values, -np.inf)
    value = float(masked_q.max())

    exp_q = np.exp(masked_q - masked_q.max())  # -inf entries exponentiate to 0
    probs = exp_q / exp_q.sum()

    return probs, value, q_values


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


def draw_viz_panel(screen, obs, placements, probs, chosen_idx, state_val,
                   font_lg, font_md, font_sm, px):

    pygame.draw.rect(screen, BG_PANEL, (GAME_W, 0, VIZ_W, TOTAL_H))
    W = VIZ_W - 20   # usable content width
    cy = 10

    # Value estimate: max masked Q (DQN has no separate critic head)
    t    = min(1.0, max(0.0, (state_val + 150) / 650))
    vcol = lerp_color(COL_BAD, COL_GOOD, t)
    screen.blit(font_lg.render(f"V  {state_val:+.1f}", True, vcol), (px, cy))
    cy += 38
    screen.blit(font_sm.render("max-Q value estimate  (red = bad board, green = good)", True, COL_DIM), (px, cy))
    cy += 20

    # Entropy of the softmax-over-Q visualization (NOT a real policy
    # distribution -- DQN always argmaxes -- this is just "how close
    # was the runner-up placement's Q-value to the winner's".
    p_valid = probs[:len(placements)]
    p_valid = p_valid[p_valid > 1e-9]
    entropy = float(-np.sum(p_valid * np.log(p_valid)))
    ecol    = lerp_color(COL_GOOD, COL_BAD, min(1.0, entropy / 3.0))
    screen.blit(font_md.render(f"H  {entropy:.2f} nats", True, ecol), (px, cy))
    cy += 26
    screen.blit(font_sm.render("Q-value spread  (↓ clear winner  ↑ close call)", True, COL_DIM), (px, cy))
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
        screen.blit(font_sm.render(f"{marker} rot={rot}  col={col:+d}", True, c), (px, cy))
        pct_surf = font_sm.render(f"{prob * 100:.1f}%", True, c)
        screen.blit(pct_surf, (px + W - pct_surf.get_width(), cy))

        # proportion bar
        bar_len = int((prob / best_p) * (W - 70))
        bar_col = COL_CHOSEN if is_chosen else lerp_color(COL_DIM, COL_ACCENT, prob / best_p)
        pygame.draw.rect(screen, bar_col, (px, cy + 14, bar_len, 4))
        cy += 26

    pygame.draw.line(screen, COL_DIM, (px, cy), (px + W, cy))
    cy += 10

        # Observation feature bars
    cy = draw_section_title(screen, "OBSERVATION FEATURES", font_sm, px, cy)

    # scalar board metrics
    screen.blit(
        font_sm.render(f"Bumpiness: {obs[20]:.3f}", True, COL_TEXT),
        (px, cy)
    )
    cy += 18

    screen.blit(
        font_sm.render(f"Aggregate Height: {obs[21]:.3f}", True, COL_TEXT),
        (px, cy)
    )
    cy += 24

    groups = [
        ("Column heights [0-9]", obs[0:10], COL_WARN),
        ("Column holes [10-19]", obs[10:20], COL_BAD),
        ("Current piece [22-26]", obs[22:27], (190, 95, 255)),
        ("Next piece [27-31]", obs[27:32], (255, 145, 45)),
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

    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n
    q_net     = QNetwork(obs_dim, n_actions).to(device)
    q_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    q_net.eval()

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
        obs, info  = env.reset()
        masks      = info["action_mask"]
        placements = env.cached_placements
        done       = False
        pygame.display.set_caption(f"Tetris AI Visualizer — Game {game_number}")

        while not done and running:
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

            # ── Q-network inference (before we step) ───────────────────────
            # Run on the CURRENT obs so the heatmap/panel show the reasoning
            # behind the piece that's about to be placed.
            probs, val, q_values = get_q_info(q_net, obs, masks, device)
            action     = int(np.argmax(np.where(masks, q_values, -np.inf)))
            chosen_idx = action  # masking guarantees this already indexes `placements`

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
            draw_viz_panel(screen, obs, placements, probs, chosen_idx, val,
                           font_lg, font_md, font_sm, PX)

            pygame.display.flip()
            clk.tick(60)
            time.sleep(STEP_DELAY)

            # ── Step ──────────────────────────────────────────────────────────
            obs, reward, done, truncated, info = env.step(action)
            masks      = info["action_mask"]
            placements = env.cached_placements

        if done:
            print(f"Game {game_number:3d} — Score: {env.game.score:5d} | Lines: {env.game.lines:3d}")
            time.sleep(1.0)

    pygame.quit()
    print(f"Stopped after {game_number} games.")


if __name__ == "__main__":
    main()
