# Tetris AI

[![example](https://img.youtube.com/vi/yYLHO8jdvUs/0.jpg)](https://youtu.be/yYLHO8jdvUs)

A Tetris implementation paired with a Double DQN agent that learns to play by evaluating afterstates, i.e., it scores every possible final resting position for the current piece, rather than picking from a fixed action space of moves.

## Overview

Instead of learning Q(s, a) over the usual `{left, right, rotate, drop}` action space, the agent enumerates every legal `(rotation, x_position)` placement for the current piece, simulates the resulting board for each one, and learns a single state value function `V(afterstate)`. At each turn it picks the placement whose simulated outcome scores highest:

```
score = reward(afterstate) + γ * V(afterstate) * (1 - done)
```

## How it works

- **Candidate generation** (`tetris_wrapper.py`): for every rotation of the current piece, every valid column, the wrapper hard drops a simulated copy of the piece and records the resulting board.
- **Feature engineering**: each candidate afterstate is encoded as a fixed size feature vector: per column heights, per column hole counts, bumpiness, aggregate height, well depth, a one-hot of lines cleared, and one-hot encodings of the current and next piece.
- **Reward shaping**: reward combines line clear bonus, a small survival bonus, and penalties/bonuses for the change in holes, height, and bumpiness caused by the placement, plus a well depth penalty and a large game over penalty.
- **Double DQN**: the online network picks the best next afterstate (masked to only valid candidates), and the frozen target network evaluates it.
- **Prioritized Experience Replay** (`PER.py` + `sum_tree.py`): transitions are sampled proportional to TD-error magnitude via a sum tree, with importance sampling weights to correct the resulting bias.
- **Variable size action sets**: since the number of legal placements changes every step, next state candidates are zero padded to `MAX_PLACEMENTS` and masked out during target computation.

## Installation

```bash
git clone <repo-url>
cd <repo>
pip install -r requirements.txt
```

Requires Python 3, `pygame`, `torch`, and `numpy`.

## Usage

### Train

```bash
cd src
python train_dqn.py
```

### Watch the trained agent play

```bash
python play.py
```

### Watch with the live diagnostics panel

```bash
python visualize.py
```

This opens an extra side panel showing:

- `V(afterstate)` for the chosen placement
- entropy of the softmax over candidate scores (a "how close was this call" indicator — the agent itself always argmaxes, this is just for visualization)
- ranked list of top candidate placements with their relative scores
- a heatmap of every candidate placement overlaid on the board (green = chosen, red→blue = score gradient)
- bar charts of the chosen afterstate's raw features (column heights, holes, piece one-hots)

Press `↑` / `↓` while it's running to speed up / slow down playback.

## Notes

- The `visualize.py` was built with AI (noted in-file).
- The network architecture is a small 2-hidden-layer MLP (`DQN_HIDDEN_SIZE = 256`).

## References

Big help from these:

- [Gonkee Video on Reinforcement Learning](https://youtu.be/VnpRp7ZglfA?)
- [Johnny Code Youtube Channel](https://www.youtube.com/@johnnycode)
- [Playing Tetris with Deep Reinforcement Learning](https://cs231n.stanford.edu/reports/2016/pdfs/121_Report.pdf)
- [Policy invariance under reward transormations by Andrew Y. Ng](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- [Tetris in Pygame](https://www.techwithtim.net/tutorials/game-development-with-python/tetris-pygame/tutorial-1)
