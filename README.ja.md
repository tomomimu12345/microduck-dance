# microduck-dance

[English](README.md) | 日本語

曲を1つ与えると、MuJoCo シミュレータの中で microduck（Pollen Robotics の小型二足ロボット）がその曲に合わせて踊る。Physical AI 応用1講座の最終課題である。

[![動画](https://img.youtube.com/vi/scfU_0LVZqI/maxresdefault.jpg)](https://youtu.be/scfU_0LVZqI)

- 説明、仕組み、再現手順: [dance/README.ja.md](dance/README.ja.md)
- 結果と経緯: 提出した説明資料（PDF）。レポートの原稿と実験記録（`dance/docs/`）は git に入れていない
- 出典と権利: [dance/data/CREDITS.md](dance/data/CREDITS.md)

## 構成

```
microduck-dance/
├── third_party/microduck-playground   上流のプロジェクト（git のサブモジュール、commit f010ef1、変更なし）
├── src/microduck_dance/               自作のパッケージ（リターゲットなど）
│   └── tasks/                         ダンスのタスクの cfg と登録
├── scripts/                           追加学習と外部の policy の取得
├── dance/                             パイプラインのスクリプト、データ、ドキュメント
└── tests/                             このリポジトリのテスト（CPU）
```

上流のファイルは1つも変更していない。上流の環境、機体モデル、タスク基盤をサブモジュールから import し、ダンスのタスクを mjlab のタスク登録に足している。どれがどちらにあるかは [dance/README.ja.md](dance/README.ja.md) の「このリポジトリとサブモジュール」にある。

## 準備

```bash
git clone --recurse-submodules https://github.com/tomomimu12345/microduck-dance
cd microduck-dance
uv sync                               # .venv を作る。サブモジュールの mjlab_microduck も入る
sudo apt install ffmpeg fonts-ipaexfont # 動画の組み立てと、そのテストに要る
bash scripts/fetch_pollen_policies.sh # roller.onnx と芸の policy（logs/tricks/ へ）
bash scripts/fetch_basketball.sh      # ボール上の b11（logs/rsl_rl/basketball/ へ）
uv run pytest                         # CPU のテスト
```

Python 3.12 と uv が要る。学習と評価には CUDA の GPU が要る。説明資料の PDF を作るときは、Windows の Arial と BIZ UDGothic を使う（`/mnt/c/Windows/Fonts` にないときは `FONTDIR` で場所を指定する）。曲ファイルは git に入っていないので、[dance/data/CREDITS.md](dance/data/CREDITS.md) の出典から入手する。

## タスク

`pyproject.toml` が `microduck_dance.tasks` を mjlab の entry point `mjlab.tasks` に登録しているので、mjlab のコマンドがそのままダンスのタスクを見つける。

```bash
uv run list-envs | grep -i dance
uv run train Mjlab-DanceV3-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5   # 短い確認
uv run python dance/scripts/evaluate/play_dance.py --checkpoint <model.pt> \
    --commands dance/data/commands/songs/swag.npz --eval --seeds 0 --num-envs 16 --json out/swag.json
```

| タスク | 内容 |
|---|---|
| `Mjlab-Dance-Flat-MicroDuck`、`Mjlab-DanceLib-Flat-MicroDuck` | 最初の版のダンスのタスク |
| `Mjlab-DanceV3-Flat-MicroDuck` | 平らな床のダンス（dance_v4 の学習に使った） |
| `Mjlab-DanceV3-Rough-MicroDuck`、`Mjlab-DanceV3-RoughMix-MicroDuck` | 凸凹の地面と、平面と凸凹を混ぜた床（dance_rough） |
| `Mjlab-BallDance-MicroDuck` | ボールの上で踊る |
| `Mjlab-RollerDance-MicroDuck` | ローラースケートで踊る（roller_dance） |

学習のログは `logs/rsl_rl/`、評価と録画の出力は `out/` に書かれる。どちらも git には入らない。学習と評価の手順の全体は [dance/README.ja.md](dance/README.ja.md) の「再現手順」にある。

## ライセンス

このリポジトリのコードは Apache License 2.0（[LICENSE](LICENSE)）である。`src/microduck_dance/basketball_finetune.py` と `scripts/finetune_basketball.py` は microduck-playground の同名のファイル（Apache-2.0）から派生したもので、各ファイルの冒頭に元のファイルと変更点を書いた（[NOTICE](NOTICE)）。

次のものは Apache-2.0 の対象外で、それぞれの条件に従う。

- 生成したダンス（`dance/data/motions/` の pkl）と、そこから作ったコマンドと学習用ライブラリ（`dance/data/commands/`、`dance/data/library.npz`）。OpenAI Jukebox（OpenAI Noncommercial Use License）と AIST++ / AIST Dance Video Database の条件により、商用に使えない。
- Microduck の3Dメッシュ（サブモジュールの中、Pollen Robotics、CC BY-NC-SA 4.0）。
- サブモジュールの中身は microduck-playground のライセンス（ソフトウェアは Apache-2.0、ハードウェアの設計ファイルは CC BY-NC-SA 4.0）に従う。

曲ファイル、EDGE と Jukebox の重み、学習したチェックポイントはリポジトリに含めない。詳細は [dance/data/CREDITS.md](dance/data/CREDITS.md) にある。
