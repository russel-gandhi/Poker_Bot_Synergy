# Discard Hold'em Bot: Team Russel & Daksh

Final submission for the Pokerbots competition (deadline 15 November 2026, 23:59 IST).

## How to run

`bot.py` is at the top level of the zip and is the entry point (`python3.11 bot.py`). It uses
the standard stdin/stdout protocol from the Protocol Specification and needs no arguments or
setup. It finds its data files through its own location, so it runs from any working directory.

## Requirements

- Python 3.11
- numpy only (no scipy, no other packages, nothing installed at runtime)
- Tested on Python 3.11.15 with numpy 1.26.4 and numpy 2.4.6, and also on Python 3.10, 3.12 and 3.13

## Contents

| File | Purpose |
|---|---|
| `bot.py` | Entry point: protocol loop, time-budget control, safe fallback replies |
| `pokerbot/core.py` | Vectorised 7-card hand evaluator and Monte Carlo equity estimates |
| `pokerbot/agent.py` | Policy network, opponent statistics and discard choice |
| `pokerbot/__init__.py` | Package marker |
| `model.json` | Trained policy weights (small neural network, about 16 KB) |
| `preflop.json` | Precomputed preflop equity table for the 1,755 suit-distinct 3-card starting hands |
| `README.md` | This file |

## How the bot plays

- **Equity:** each decision estimates win probability by Monte Carlo sampling. It estimates
  against a random hand and against the stronger part of the opponent's possible range, using
  the cards exposed by the discard.
- **Policy:** a small neural network chooses between fold, check/call, three raise sizes
  (0.4, 0.8 and 1.5 pot) and all-in. Its inputs are the equity estimates, pot odds, stack-to-pot
  ratio, street, position and simple per-match statistics on the opponent.
- **Discard:** the bot scores each of the three possible two-card keeps by Monte Carlo
  equity on the current board. A small learned adjustment rewards pairs, suited cards and
  connected cards, and the bot discards the card left out of the best-scoring keep.
- **Training:** reinforcement learning in a population-based "battle royale" of self-play
  (elimination, mutation and a hall of fame). It started from a behaviour-cloned heuristic and
  was selected on its worst-case results against a fixed panel of earlier versions and
  built-in styles. All training used our own simulator and our own bots.

## Compliance with the rulebook

- **Section 5.4:** no network access, no subprocesses, no file writes. The bot only reads
  `model.json` and `preflop.json` from its own folder.
- **Section 5.8:** stdout carries only protocol JSON, and every reply is flushed. Diagnostics
  go to stderr only if an internal error occurs, and in normal play stderr stays empty.
- **Time bank:** the bot reads `time_left_ms` on every request and reduces its Monte Carlo
  sample count as the bank shrinks. Below 1.5 s left it replies instantly with a safe legal
  action. In testing it used at most about 13 s of the 75 s Stage 2 bank, and every reply
  came in well under the 5 s limit.
- **Resources:** about 33 MB peak memory. BLAS threads are set to 1 before numpy is
  imported, so the bot runs comfortably on 1 vCPU and 1 GB RAM.
- **Robustness:** any internal error falls back to a legal reply instead of crashing.
  Unknown message types and fields are ignored. The bot exits after `match_end`.
- **Section 5.9:** the bot keeps no state between legs or matches.
