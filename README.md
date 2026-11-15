# Poker Bot Synergy: Discard Hold'em Bot (Team Russel & Daksh)

A heads-up no-limit **Discard Hold'em** bot for the Pokerbots competition. It combines live Monte
Carlo equity estimates with a small neural-network policy, trained by reinforcement learning in a
population-based "battle royale".

## Final submission

**`final_submission_Seven_Duce.zip`** is the bot to upload. Rebuild it from source with
`python -m train.export checkpoints/champion_v4.json --out submission`.

```
final_submission_Seven_Duce.zip
├── bot.py            # entry point (stdin/stdout JSON protocol)
├── README.md         # judges' guide: how to run, contents, compliance
├── model.json        # trained policy weights (v4 champion)
├── preflop.json      # preflop equity table (1,755 suit-distinct 3-card hands)
└── pokerbot/
    ├── __init__.py
    ├── agent.py      # policy network, opponent model, discard logic
    └── core.py       # vectorised hand evaluator + Monte Carlo equity
```

Run it with `python3.11 bot.py` (Python 3.11, numpy only). The judges' README inside the zip
covers usage and rulebook compliance in full. `submission_v3_fallback.zip` is the previous
champion, kept as a backup.

## How the bot plays

- **Equity:** each decision estimates win probability by Monte Carlo sampling. It estimates
  against a random hand and against the stronger part of the opponent's range, using the cards
  exposed by the discard.
- **Policy:** a small neural network picks fold, check/call, one of three raise sizes
  (0.4, 0.8 or 1.5 pot) or all-in. Its inputs are equity estimates, pot odds, the
  stack-to-pot ratio, street, position and per-match opponent statistics.
- **Discard:** the bot scores each two-card keep by equity, with a small learned adjustment
  for pairs, suited cards and connected cards.
- **Safety:** the Monte Carlo sample count adapts to the time bank. Any error falls back to a
  safe legal reply, and stdout carries only protocol JSON.

## How it was trained

1. **Behaviour-cloning warm start** (`train/warmstart.py`): fit the network to a hand-written
   equity heuristic, so training starts from a sensible player.
2. **Battle royale** (`train/battle_royale.py`): a population of policies plays duplicate-deal
   matches in a rule-exact simulator (`pokerbot/sim.py`). Each generation the weakest half is
   eliminated, and survivors are mutated and crossed over. REINFORCE uses a duplicate baseline,
   and a hall of fame keeps past champions in the field.
3. **Robust fitness and selection** (`train/robust_select.py`): candidates are ranked by
   worst-case results against a fixed panel of earlier champions and built-in playing styles.
   Ranking by the total score against siblings is not used. This avoids rock-paper-scissors
   regressions.
4. **Testing** (`train/test_bot.py`, `train/h2h.py`): every candidate plays real matches on
   the organisers' official engine, Stage 1 and Stage 2, before it can be promoted.

| Version | What changed | Outcome |
|---|---|---|
| v1 | First battle-royale champion | Baseline |
| v2 | Range-aware equity features | Regressed against v1 (league non-transitivity) |
| v3 | Robust worst-case fitness and selection | Outscored v1 in held-out official-engine tests |
| **v4** | 300 more generations seeded from v3, with v3 in the fixed panel | **Beat v3 14/20** on the official engine (both seat orders, both stages). Now the submission |

`RESEARCH.md` has the full strategy research and design notes.

## Repository layout

| Path | Contents |
|---|---|
| `pokerbot/` | Evaluator, equity, agent, simulator, deal-pool builder, genome file I/O (training only) |
| `train/` | Warm start, battle royale, selection, export, official-engine tests, head-to-head |
| `deploy/bot.py` | Bot template packaged into each submission |
| `arena/` | Simple baseline bots (station, maniac, nit, equity, random) |
| `tests/` | Evaluator tests and simulator-vs-official-engine parity tests |
| `checkpoints/` | Trained genomes (`champion_v4.json` is the submitted model) |
| `results_*/`, `*_first.txt` | Official-engine match results from evaluation runs |
| `run_pipeline*.sh` | End-to-end training pipelines for each version |
| `Starter_Kit/` | Organisers' engine and starter bot (needed for testing) |

## Reproducing

```
python -m pokerbot.dealpool --out data/pool2.npz                                     # deal pool + preflop table
python -m train.warmstart --pool data/pool2.npz --out checkpoints/bc.json            # BC warm start
bash run_pipeline_v4.sh                                                              # train, select, export, test
python -m train.test_bot --bot submission_v4/bot.py --python python3.11              # official-engine checks
```

Training uses numpy and multiprocessing only. On a 16-core machine one generation takes
about 30-40 s.

## Notes

- Tested on Python 3.10, 3.11 (numpy 1.26 and 2.4), 3.12 and 3.13.
