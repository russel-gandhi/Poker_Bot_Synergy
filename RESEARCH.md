# Discard Hold'em Pokerbot: Research, Strategy and Training Plan

Sources: `problem_statement.pdf`, `rulebook.pdf`, `protocol_specification.pdf`, `Starter_Kit/`
(engine.py, bot.py, random_bot.py, README). Deadline: **15 Nov 2026, 23:59 IST**. Code freezes then.

---

## 1. What the documents actually reward (read this first)

| Fact from the rules | What it means for the bot |
|---|---|
| A match is won on **net chips over both legs**. Points are win 1 / draw 0.5 / loss 0 (RB 8.1-8.2). | You are maximising **P(win the match)**, not raw chip EV. A small, reliable edge beats a high-variance one. |
| Tie-break is **capped** chips per hand, C = 400 (Stage 1) / 1000 (Stage 2) (RB 8.3). | Beating a weak bot by 3,000 chips is worth the same as +400. Once far ahead, cutting variance costs almost nothing. |
| **Duplicate deals**: every deal is played twice with cards and seats swapped (PS 2.3, RB 7.1). | Card luck mostly cancels. **Decision quality** decides matches, so even small leaks show up. |
| A crash, timeout, 5 s overrun or wall-clock kill **forfeits the match** at -C (RB 9.1-9.2). | Robustness comes first. One exception equals one lost match at maximum tie-break penalty. |
| A debug print on stdout becomes your reply. Every later action is then check/fold (PS 8.6). | Only protocol JSON goes to stdout. Log to stderr (64 KB cap). |
| Illegal action becomes check, or fold if check is illegal. Illegal discard drops the lowest card (RB 9.3-9.4). | Always clamp raise amounts to `[min_raise, max_raise]` and choose only from `legal_actions`. |
| Time bank is 30 s per 200 hands (Stage 1) and 75 s per 500 (Stage 2), so **~150 ms per hand**. 5 s per reply. 1 vCPU, 1 GB. | About 25-35 ms per decision. Monte Carlo equity has to be vectorised (numpy) and cached per hand. |
| The first 5 s after launch are **not charged** (RB 5.7). | Load or build lookup tables at start-up for free. |
| Python 3.11, **stdlib + numpy + scipy only**, ≤ 5 MB, read-only filesystem. | Train with anything you like (torch etc.), but **inference must be numpy**. Ship weights and tables as JSON/npz inside the zip. |
| State lasts only within a leg. Identifying duplicate hands or carrying info between legs is **prohibited** (RB 10.3). | Opponent modelling must re-learn from scratch each leg (200 or 500 hands). Never try to detect duplicates. |
| Stacks reset to 200 (100 BB) every hand. **Max 4 bets/raises per street**. | Every hand is a fresh 100 BB game. Pre-flop has at most 3 voluntary raises, and at the cap you can only call or fold. |
| Code similarity checks. Copying public bot repos is banned. Published algorithms are allowed (RB 10.1, 10.4). | Implement CFR/RL yourselves (fine). Do not paste open-source poker bots. |
| Stage 1 is a Swiss of 6-8 rounds (round robin if < 64 teams). The top 32 go to a Stage 2 round robin of 31 matches each. | Stage 1 is mostly against weak or buggy bots, so **exploitation** pays. Stage 2 is against competent bots, so **robustness / low exploitability** pays. |

## 2. How the variant changes poker

1. **Three hole cards, then keep two.** There are 22,100 starting hands, which reduce to **1,755 suit-isomorphic classes** (computed in `data/preflop.json`). You can choose your best 2-of-3, so average hand strength is higher than in Hold'em:
   * P(a pair among 3 hole cards) ≈ 17%, versus 5.9% in Hold'em. Suited and connected pairs are also far more common.
   * **Trips in the hand are the worst hands** (`2s 2h 2d` has equity 0.29). You must throw one away, which kills your own outs.
   * The best hands are `AAx`, especially AA plus a suited kicker (`As Ah Qs` ≈ 0.76 equity versus a random 3-card hand).
   * Showdown values shift up. Top pair is worth less than in Hold'em, and two pair or better and nut draws matter more.
2. **The discard happens after flop betting, simultaneously and face up.**
   * You pick the keep-pair with the most equity on this flop. Our bot ranks the three keeps by Monte Carlo equity, plus learned adjustments.
   * **The opponent's discard is public information.** If they throw a K on a low board, they kept two cards they valued above a king (pair, set or draw). If they throw a low card, the kept cards are probably high or connected. Our features include the opponent's discard rank.
   * **Your discard leaks information.** A deliberately "deceptive" discard (giving up a little equity to hide a draw) is a legitimate mixed-strategy idea. The trainer can learn this through the discard weights.
   * **Card removal:** both discards are dead cards. Use them when counting outs.
   * Flop betting happens **with 3 cards**, before anyone knows the discards. Hands with three-way potential (e.g. two flush draws plus a pair) are much stronger on the flop than they look.
3. **The discard happens even when all-in.** After a pre-flop all-in you still get a `discard_request`, so always answer it.
4. **Position.** The button acts first pre-flop and last post-flop (standard heads-up). Positional aggression applies.
5. **Raise cap of 4.** Pre-flop: BB, raise, 3-bet, 4-bet and then only call or fold. A 4-bet to about 50 caps the pot at about 100. Escalation is bounded, which makes shove/fold trees small.

## 3. State of the art: what "best poker bot" means

| Approach | Landmark systems | Strength | Fit for this contest (4 days, Python, 150 ms/hand) |
|---|---|---|---|
| **CFR / MCCFR + abstraction** (blueprint) | Cepheus (limit, near-solved), Libratus (HUNL 2017), Pluribus (6-max 2019) | Near-equilibrium, hard to exploit | Strong but compute-heavy. The 3-card + discard tree needs new abstractions. A *coarse* blueprint (preflop + bucketed postflop) is feasible in a few days. |
| **Real-time subgame solving / depth-limited search** | DeepStack, Libratus endgame solver, ReBeL, Student of Games | Fixes blueprint errors live | Too slow in pure Python within 30 ms per decision. Only a tiny river re-solve is plausible. |
| **Deep CFR / NFSP / PPO self-play** | Deep CFR (2019), NFSP (2016), various PPO poker agents | Learns from scratch; no hand-made abstraction | Pure self-play **cycles** (rock-paper-scissors) and becomes exploitable. It needs a league or population. Inference must be numpy (fine for a small MLP). |
| **League / Population-Based Training** | AlphaStar league, PBT (DeepMind) | Robust against diverse opponents; combats cycling | **Our choice.** It also lets us score training exactly like the real tournament. |
| **Evolution strategies** | OpenAI-ES (2017) | Gradient-free RL; robust to noisy rewards | Used for weights *and* hyper-parameters inside the battle royale. |
| **Opponent modelling / exploitation** | Vexbot, Polaris-era exploiters, Bayesian opponent models | Beats weak fields by large margins | Essential in Stage 1, where most opponents are simple bots. |
| **Equity-threshold heuristics** | Most hackathon winners | Fast, robust baseline | It is our warm start and one of the gauntlet opponents. |

**Verdict:** the strongest thing achievable in 4 days is a **hybrid**: equity features from a fast vectorised Monte Carlo engine, an MLP policy over abstract actions, opponent-model features, and training as an **RL league (battle royale)** against itself, past champions and fixed exploitable styles. A CFR blueprint would be stronger in theory, but there isn't time to build a correct abstraction for a new variant before the deadline.

## 4. The complete strategy list

### 4.1 Robustness (non-negotiable; each item protects a whole match)
1. Wrap every decision in `try/except`. On error, fall back to check, then a cheap call, then fold (`deploy/bot.py: safe_action`).
2. Never print to stdout except replies. Flush every reply.
3. Clamp raise amounts to `[min_raise, max_raise]`. Pick only from `legal_actions` (raise can disappear when the opponent is all-in).
4. Always answer `discard_request`, including all-in hands.
5. Budget time from `time_left_ms`. Scale Monte Carlo samples 256 → 128 → 48 → 16. Below 1.5 s left, switch to instant heuristic replies.
6. Exit on `match_end`. Ignore unknown message types and fields.
7. Test with the official `engine.py` at Stage 1 and Stage 2 lengths with **zero notices** (`train/test_bot.py`).

### 4.2 Hand strength and card play
8. **Preflop:** a 1,755-class lookup table of 3-card equity (vs a random 3-card hand, both keeping their best 2 at showdown).
9. **Flop (3 cards):** equity of the full 3-card hand versus a random 3-card opponent.
10. **Discard:** evaluate all 3 keep-pairs (the dropped card is dead) and keep the best. A learned bonus for pairs, suitedness, connectedness and high cards gives room for deception.
11. **Turn/river:** equity of the kept pair against a random 2-card opponent, with your own discard dead.
12. **Pot odds:** call when equity exceeds `to_call / (pot + to_call)` plus a margin that depends on how aggressive the opponent has been.
13. **Bet sizing:** abstract sizes of 0.4, 0.8 and 1.5 pot, plus all-in. These cover the useful range. Small bets for thin value and bluffs, large bets for polarised value.
14. **Board texture:** flush boards, paired boards and opponent discard rank are all policy inputs.

### 4.3 Exploitation (within a leg, from scratch)
15. **Fold-to-bet rate:** bluff more against bots that fold too much. Many student bots fold to any bet without a strong hand.
16. **Aggression rate:** against maniacs, call down lighter and stop bluffing. Against nits, steal blinds and fold to their raises.
17. **VPIP:** separates calling stations from tight players. Against stations, never bluff and value-bet thinner and bigger.
18. Bayesian-smoothed stats (priors), so the first 20-30 hands don't overreact.
19. **Starter-kit bots are certain opponents.** The check/call `bot.py` is a pure calling station, and `random_bot.py` folds 20% of the time when facing a bet. The trainer's gauntlet includes both styles.

### 4.4 Match-level (scoring-aware) play
20. A win is worth the same whether by 1 chip or 5,000, so prefer **lower-variance lines** when the edge is similar.
21. **Lead protection:** if this leg's net is already far above the cap, tighten up (fewer marginal all-ins). This is optional and not yet implemented. Note that you can't see the other leg, so be conservative.
22. Mixed strategies: actions are sampled from a softmax with a learned temperature, so the bot is harder to read and exploit.

### 4.5 Ideas for after the first submission
23. River **exact equity** (enumerate all ~990 opponent combos) instead of Monte Carlo.
24. **Range-aware equity:** weight opponent holdings by the actions they took and by the discard they made (they kept their best 2 of 3).
25. A small **CFR blueprint** for preflop (1,755 classes x the action tree is small) to replace the learned preflop.
26. Shove/fold charts for short SPR spots after big preflop pots.

## 5. The Battle Royale RL training process (implemented)

```
             ┌──────────────── each generation ────────────────┐
 pop (24) ──►│ 1. fresh 200-deal set (shared by every match)   │
 + 5 fixed   │ 2. full round robin, duplicate matches          │
 baselines   │    (pop × pop) + (pop × baselines) + (pop × HoF)│
 + Hall of   │ 3. score = match points, tie-break capped       │
 Fame (≤8)   │    chips/hand  ← identical to the real rulebook │
             │ 4. RL: REINFORCE step for every agent           │
             │    advantage = duplicate chip total of the deal │
             │ 5. ELIMINATE bottom 50%                         │
             │ 6. survivors breed: crossover + self-adaptive   │
             │    mutation of weights, lr, sigma, temperature  │
             │ 7. every 5 gens: champion → Hall of Fame        │
             └─────────────────────────────────────────────────┘
 after N gens: GRAND FINAL (Stage-2 format, 500 deals × 3 deal sets, top 8 + all HoF + baselines)
            → winner → export → TESTING with the official engine.py (real processes and clocks)
```

* **Why RL with a league instead of pure self-play:** pure self-play poker agents cycle and overfit to their own quirks. Fixed exploitable baselines force the agent to *exploit*. The Hall of Fame forces it not to forget. The population keeps strategies diverse.
* **RL signal:** REINFORCE with the **duplicate baseline**. Each decision on a deal is credited with (our chips on those cards) minus (the opponent's chips on the same cards). This is the same variance reduction the organisers use, applied to the gradient.
* **Outer loop:** population-based training / evolution. Hyper-parameters evolve along with the weights. Fitness is the real tournament score, so we optimise what is actually measured.
* **Warm start:** behaviour cloning of the equity heuristic (95% action agreement), as in AlphaStar's supervised-then-RL pipeline.
* **Train/deploy parity:** the same `pokerbot/agent.py` runs in training and in the submission. Training uses precomputed equities from `data/pool.npz`, computed with the same estimator the live bot uses. `tests/test_parity.py` shows the in-process sim matches the official `engine.py` chip for chip.

### Files
| Path | Purpose |
|---|---|
| `pokerbot/core.py` | Vectorised 7-card evaluator (validated against engine.py on 20k hands), Monte Carlo equity, suit isomorphism |
| `pokerbot/agent.py` | Features, opponent model, MLP policy, discard logic, live oracle |
| `pokerbot/baselines.py` | Gauntlet: random, calling station, maniac, nit, equity (TAG) |
| `pokerbot/sim.py` | Fast in-process engine (rule-exact), duplicate matches |
| `pokerbot/dealpool.py` | Preflop table + deal pool with precomputed equities |
| `train/warmstart.py` | Behaviour cloning warm start |
| `train/battle_royale.py` | The league / battle royale trainer (resumable) |
| `train/grand_final.py` | Stage-2-format final among the survivors and the Hall of Fame |
| `train/export.py` | Builds `submission/` and `submission.zip` |
| `train/test_bot.py` | Official-engine tests: every opponent, Stage 1 and Stage 2 timing, zero notices |
| `run_pipeline.sh` | Runs all of the above in order |

### Commands
```bash
./run_pipeline.sh                                   # everything, end to end
python -m train.battle_royale --name br2 --pop 32 --gens 300 --resume   # longer run
python -m train.grand_final --run runs/br2
python -m train.export runs/br2/final_champion.json --out submission
python -m train.test_bot --bot submission/bot.py
```

## 6. Four-day timeline
| Day | Goal |
|---|---|
| Day 1 (11 Nov) | Pipeline built and verified; first run (`br1`); **upload a first valid submission immediately** (only the last valid upload counts, so upload early and often). |
| Day 2 | Longer run (pop 32, 300+ gens). Add range-aware equity (4.5 #24) and exact river equity. Compare against br1 in the grand final. |
| Day 3 | Stage 2 timing tests on a slow machine. Opponent-model tuning. Second upload. |
| Day 4 (15 Nov) | Freeze by the evening. Final official-engine test with a Python 3.11 interpreter. Upload before 23:59 IST. |

## 7. Risks
* **Python version:** the target is 3.11 and this machine has 3.10/3.12/3.13. The bot code avoids version-specific syntax and has been tested on 3.10 with the official engine. If possible, also run the final check under 3.11.
* **Speed on the grader:** a 1-vCPU grader may be slower than this PC. Adaptive sampling keeps us safe, and `test_bot` flags any run using > 60% of the bank.
* **Overfitting to the gauntlet:** real opponents differ. The Hall of Fame plus population diversity mitigates this, and the grand final uses fresh deals.
