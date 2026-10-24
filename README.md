# microduck-dance

English | [日本語](README.ja.md)

Give it a song, and microduck (a small bipedal robot by Pollen Robotics) dances to it in the MuJoCo simulator. The dance comes from EDGE, a retarget turns it into robot commands, and RL policies trained with mjlab on top of [microduck-playground](https://github.com/Vottivott/microduck-playground) follow them on a flat floor, on rough terrain and on roller skates. This is the final assignment of the Physical AI 応用1 course.

[![Video](https://img.youtube.com/vi/scfU_0LVZqI/maxresdefault.jpg)](https://youtu.be/scfU_0LVZqI)

- How it works and how to reproduce it: [dance/README.md](dance/README.md) (in Japanese)
- Results and the path to them: the submitted report (PDF). The report source and the experiment log (`dance/docs/`) are not in git
- Sources and rights: [dance/data/CREDITS.md](dance/data/CREDITS.md)

## Layout

```
microduck-dance/
├── third_party/microduck-playground   upstream project (git submodule, commit f010ef1, unmodified)
├── src/microduck_dance/               own package (retarget and more)
│   └── tasks/                         cfgs and registration of the dance tasks
├── scripts/                           extra training and fetching external policies
├── dance/                             pipeline scripts, data and documentation
└── tests/                             tests of this repository (CPU)
```

No upstream file is modified. The environments, robot model and task framework are imported from the submodule, and the dance tasks are added to the mjlab task registry. The section 「このリポジトリとサブモジュール」 in [dance/README.md](dance/README.md) lists which part lives where.

## Setup

```bash
git clone --recurse-submodules https://github.com/tomomimu12345/microduck-dance
cd microduck-dance
uv sync                               # creates .venv, including mjlab_microduck from the submodule
sudo apt install ffmpeg fonts-ipaexfont # needed to build the video and for its tests
bash scripts/fetch_pollen_policies.sh # roller.onnx and the trick policies (into logs/tricks/)
bash scripts/fetch_basketball.sh      # b11 on the ball (into logs/rsl_rl/basketball/)
uv run pytest                         # CPU tests
```

Python 3.12 and uv are required. Training and evaluation need a CUDA GPU. Building the report PDF uses Arial and BIZ UDGothic from Windows (set `FONTDIR` if they are not in `/mnt/c/Windows/Fonts`). The song files are not in git. Get them from the sources listed in [dance/data/CREDITS.md](dance/data/CREDITS.md).

## Tasks

`pyproject.toml` registers `microduck_dance.tasks` under the mjlab entry point `mjlab.tasks`, so the mjlab commands find the dance tasks directly.

```bash
uv run list-envs | grep -i dance
uv run train Mjlab-DanceV3-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5   # quick check
uv run python dance/scripts/evaluate/play_dance.py --checkpoint <model.pt> \
    --commands dance/data/commands/songs/swag.npz --eval --seeds 0 --num-envs 16 --json out/swag.json
```

| Task | Description |
|---|---|
| `Mjlab-Dance-Flat-MicroDuck`, `Mjlab-DanceLib-Flat-MicroDuck` | first version of the dance task |
| `Mjlab-DanceV3-Flat-MicroDuck` | dance on a flat floor (used to train dance_v4) |
| `Mjlab-DanceV3-Rough-MicroDuck`, `Mjlab-DanceV3-RoughMix-MicroDuck` | rough terrain, and a floor mixing flat and rough parts (dance_rough) |
| `Mjlab-BallDance-MicroDuck` | dance on a ball |
| `Mjlab-RollerDance-MicroDuck` | dance on roller skates (roller_dance) |

Training logs go to `logs/rsl_rl/`, and evaluation and recording outputs go to `out/`. Neither is in git. The full training and evaluation steps are in the section 「再現手順」 of [dance/README.md](dance/README.md).

## License

The code in this repository is under the Apache License 2.0 ([LICENSE](LICENSE)). `src/microduck_dance/basketball_finetune.py` and `scripts/finetune_basketball.py` are derived from the files of the same name in microduck-playground (Apache-2.0). The head of each file names the original and the changes ([NOTICE](NOTICE)).

The following are not covered by Apache-2.0 and follow their own terms.

- The generated dances (pkl files in `dance/data/motions/`) and the commands and training library made from them (`dance/data/commands/`, `dance/data/library.npz`). They cannot be used commercially, because of the terms of OpenAI Jukebox (OpenAI Noncommercial Use License) and of AIST++ / the AIST Dance Video Database.
- The Microduck 3D meshes (inside the submodule, Pollen Robotics, CC BY-NC-SA 4.0).
- The contents of the submodule follow the microduck-playground license (Apache-2.0 for software, CC BY-NC-SA 4.0 for the hardware design files).

Song files, the EDGE and Jukebox weights and the trained checkpoints are not included. Details are in [dance/data/CREDITS.md](dance/data/CREDITS.md).
