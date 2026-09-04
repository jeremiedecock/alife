# ALife

ALife was originally conceived as a game, then an artificial life simulation (hence the name, ALife), then a `gym`-inspired testbed for reinforcement learning agents, then a multi-player MMORPG-style game with integrated server environment for scorekeeping, then (currently) just a standalone multi-agent PettingZoo environment. The server/networking architecture and agents are now maintained in separate projects, see TODO.git and TODO.git for hosting the environment, and connecting to one with an agent, respectively.

![Screenshot](screenshot.png "Screenshot")

## Install

Install the latest released wheel, published as a GitHub Release asset:
```
pip install https://github.com/jeremiedecock/alife/releases/download/v0.1.0/alife-0.1.0-py3-none-any.whl
```

If the repository is private, download the asset with an authenticated client first:
```
gh release download v0.1.0 -R jeremiedecock/alife -p '*.whl'
pip install ./alife-*.whl
```

To work on the environment itself, install it from a clone in editable mode:
```
git clone git@github.com:jeremiedecock/alife.git
cd alife
pip install -e .
```

## Running

Quick start:
```
python run.py
```

This is a multi-agent PettingZoo environment. You can run it as per any other PettingZoo environment. 


##  'Gameplay'

Numerous gameplay configurations are possible involving e.g., speedrunning, bumper cars, capture the flag, king of the hill, etc. Involving the bugs (agents), plants, and obstacles/rocks. The observation and action space are kept as simple as possible to avoid the need for super-deep architectures. Vision is based on 2-pixel RGB-color vision, plus some 'health/energy' level and a rough sensor indicating which way it is pointing in the map.  

## The environment in detail (PettingZoo API)

`World` ([alife/aworld.py](alife/aworld.py)) implements PettingZoo's `ParallelEnv`: at each step
*all* agents act simultaneously and `step()` takes/returns dicts keyed by agent id.

```python
from alife import World
env = World(render_mode=None)          # None | "human" | "rgb_array"
env.reset(options={'map_name': "2025"})
agent_id = env.add_agent("my_bug")     # agents are created AFTER reset()
```

Two things differ from a typical PettingZoo environment and matter a lot in practice:

* **Agents are added after `reset()`**, with `add_agent(name)` (and removed with `del_agent(id)`).
  `reset()` loads the map and its inanimate sprites, and returns **empty** `observations` /
  `infos` dicts — the first real observations arrive with the first `step()`.
* **Agent ids are integers, not strings.** `add_agent()` returns the id, which is also the row
  index of that agent in the internal state array. Ids start just above the number of inanimate
  sprites (e.g. `32` on the `2025` map), so do not assume they start at `0`.

Episodes are *eternal*: `truncations` is always `False`, `env.agents` never empties by itself, and
a death does not end the episode (the bug respawns at a nest). It is up to the caller to stop the
loop after a given number of steps.

### Observations

`observation_space(agent)` is `Box(0.0, 1.0, (11,), float32)`. The vector is a low-dimensional
"sensor" reading, deliberately small so that shallow policies can learn from it:

| Index | Name | Meaning |
|---|---|---|
| 0 | `COLIDE` | `1.0` if the body is currently overlapping another sprite (rock, plant or bug), else `0.0`. |
| 1 | `PROXIMITY` | How crowded/close things are: sum of the overlaps of the outer sensing ring (`OUTER_RADIUS = 64` px) with nearby sprites, plus the proximity to the nearest wall/water pixel, clipped to `[0, 1]`. `0` = nothing within 64 px, `1` = touching. |
| 2-4 | `PROBE1` | RGB intensity seen at the tip of the **left** antenna (`+0.3` rad from the heading, ~39 px away). |
| 5-7 | `PROBE2` | RGB intensity seen at the tip of the **right** antenna (`-0.3` rad from the heading). |
| 8 | `ENERGY` | `health / MAX_HEALTH`, clipped to `[0, 1]`. A bug is born at `0.5` and dies at `~0.01`. |
| 9 | `COMPASS` | Alignment with the current target: the cosine of the angle between the heading and the direction of the target flag (`1` = pointing straight at it), forced to `0` when pointing away from it. |
| 10 | `SPEED` | Current speed normalised by `MAX_SPEED` (10 px/step). |

Vision is therefore just **two RGB pixels**, one per antenna, whose colour encodes *what kind of
thing* is at the antenna tip:

| Colour | Value | What it is |
|---|---|---|
| black | `[0, 0, 0]` | nothing (empty ground) |
| blue | `[0, 0, 1]` | a rock **or** a wall/water tile |
| green | `[0, 1, 0]` | a plant (food) |
| red | `[1, 0, 0]` | another bug |

Flags and nests are **invisible** to the antennae (and cannot be bumped into): the only information
about the target is `COMPASS`.

Those two pixels are computed geometrically from the terrain grid and the neighbour list, never read
back from the pygame surface, so **observations are exactly the same with rendering on or off**.

### Actions

`action_space(agent)` is `MultiBinary(2)`: a 2-bit array `[right, left]`, one bit per "leg". The two
bits are combined into a thrust and a turn, which drive a smooth acceleration model rather than
teleporting the bug:

| Action | Effect |
|---|---|
| `[0, 0]` | brake (target speed 0), no turn |
| `[1, 1]` | full thrust straight ahead (target speed `MAX_SPEED = 10` px/step) |
| `[1, 0]` | half thrust, turn right by `0.10` rad |
| `[0, 1]` | half thrust, turn left by `0.10` rad |

Internally: `target_speed = 0.5 * (right + left) * MAX_SPEED` and `turn_delta = (right - left) * 0.10`.
Speed then moves toward `target_speed` by `+0.2` per step when accelerating and `-0.8` per step when
braking, and the bug advances by `speed` pixels along its heading.

There is no explicit "attack" or "eat" action: the bug carries a **spear** sticking out ~32 px in
front of it, and whatever the spear tip touches is acted upon automatically (see below).

### Reward

The reward is reset to `0` at every step and only two events produce a non-zero value:

| Event | Reward |
|---|---|
| The spear tip touches the **current target flag** (checkpoint reached) | `+5` (`RWD_CHECKPOINT`) |
| Death — energy exhausted, or entering water/cliff | `-5` (`RWD_DEATH`) |

Everything else (eating, bumping, moving) has **no direct reward**; it only affects energy, and
therefore the reward indirectly, through survival. Since both events are resolved in the same step,
a reward can also be the sum of the two (e.g. reaching a checkpoint and dying gives `0`).

`terminations[agent]` is `True` exactly on the step where the agent dies; the agent is immediately
respawned at a random nest and keeps playing.

### How the world evolves

Each `step()` runs two clearly separated phases: `_update()` applies the actions and resolves all
interactions (state mutation only), then `_observe()` fills in the observation fields from the final
positions. Collision candidates come from a scipy KD-tree rebuilt every step.

**The circuit.** Flags are numbered by a sub-id: sub-id `0` marks the **nests** (spawn points), and
`1, 2, 3, …` form a **circuit of checkpoints**. Each bug tracks one target at a time; touching it
with the spear scores `+5` and the target becomes the next number in the circuit, wrapping back to a
nest after the last one. The next target is picked at random among the sprites carrying that sub-id.

**Energy.** Energy (`health`, max `100`) is the real currency of the game:

| Event | Energy change |
|---|---|
| Cost of living | `-0.01` per step |
| Spear tip on a plant (eating) | `+10` for the bug, `-10` for the plant |
| Spear tip on another bug | `-10` for the victim |
| Bumping into anything | `-5 * speed` for the bug (so crashing at full speed hurts a lot) |
| Bumping into another bug | additionally `-5` for the victim |
| Entering a wall / water tile | instant death |

When energy drops to `~0`, the bug dies (`-5`), respawns at a random nest with 50% energy, a random
heading and a random first checkpoint, and its speed is reset to `0`. Bumping also pushes the bug
back off the obstacle it hit.

So a good policy has to interleave two objectives: run the checkpoint circuit for points, and eat
enough plants (while avoiding walls, water and other bugs) to keep the energy up.

### Full example

```python
from alife import World

MAX_STEPS = 1000

env = World(render_mode="human")          # use render_mode=None to run headless
env.reset(options={'map_name': "2025"})   # always pass the map name explicitly

# Agents must be created after reset(); the returned id is the PettingZoo key
agent_ids = [env.add_agent(f"agent_{i}") for i in range(4)]

total = {i: 0.0 for i in agent_ids}

for t in range(MAX_STEPS):
    # Replace this with your policy: observations[i] -> action
    actions = {i: env.action_space(i).sample() for i in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)

    for i, r in rewards.items():
        total[i] += r
        if terminations[i]:
            print(f"[t={t}] agent {i} died and respawned")

    env.render()

print("Total rewards:", total)
env.close()
```

A minimal single-agent loop reacting to observations:

```python
import numpy as np
from alife.constants import IDX_OBS   # column names, if you need them

obs = np.zeros(11, dtype=np.float32)
while True:
    collide, proximity = obs[0], obs[1]
    green_left, green_right = obs[3], obs[6]       # plant seen by each antenna
    compass = obs[9]

    if collide or proximity > 0.8:
        action = np.array([1, 0])                  # turn away
    elif green_left > green_right:
        action = np.array([0, 1])                  # steer toward food on the left
    elif green_right > green_left:
        action = np.array([1, 0])
    elif compass > 0.9:
        action = np.array([1, 1])                  # aligned with the target: charge
    else:
        action = np.array([1, 0])                  # scan around

    observations, rewards, terminations, truncations, infos = env.step({agent_id: action})
    obs = observations[agent_id]
```

### Rendering

`render_mode` is chosen in the constructor:

* `None` — no pygame window, no rendering (fastest; observations are unchanged).
* `"human"` — opens a window and `render()` draws the frame, capped at 10 FPS.
* `"rgb_array"` — `render()` returns the frame as an `(H, W, 3)` `uint8` array.

Note that `"rgb_array"` still opens a pygame display; on a headless machine, run with
`SDL_VIDEODRIVER=dummy`.

### Known quirks

A few things are worth knowing before you spend time debugging them:

* `World.reset()` accepts a `seed` argument but **ignores it**; the environment uses the global
  `numpy.random` state (respawn positions, target selection). Seed `numpy.random` yourself if you
  need reproducibility.
* The default `map_name` of both `run.py` and `World.reset()` is `new_4`, which does not exist.
  Only `2025` and `fairy_ring` ship in [alife/maps/](alife/maps/) — always pass `map_name`
  explicitly. `ALIFE_MAP_DIR` overrides the map directory.
* The checkpoint circuit is selected purely by sub-id, without checking the sprite type, so a
  sub-id also used by a plant makes that plant a checkpoint. On the shipped maps the circuit is
  `1, 2, 3, 4, 5, 6, 7, 8` where `5` and `8` are plants, not flags.
* `COMPASS` is declared inside `Box(0, 1)` but its update rule can make it slightly negative, so an
  observation may fall marginally outside the declared observation space.
* At most `160` agents can live in the world at once (the state array holds `200` sprite slots,
  the last `40` being reserved for visual effects).

## Map Editing and Environment Design
 
To generate a new map, run 
```
python3 -m alife.map_editor 
```
and specify the name of the new map. To edit an existing map:
```
python3 -m alife.map_editor new_4.map
```

Press <kbd>h</kbd> to see the keys for editing. 

## Notes on Graphics

* Terrain and decor obtained from [Open Game Art](https://opengameart.org/users/chabull) 
<!-- http://chabull.praire-chicken.com/index.html#work --> 
<!-- Csaba Felvegi -->
* Bug graphics from [Open Clip Art](https://openclipart.org/tags/ladybug)

