# dance/scripts

パイプラインの段階ごとにフォルダを分けている。`dance/docs/`（レポートの原稿と図）は git に入れていないので、`terrain_figure.py` と `build_report_pdf.sh` はそのファイルが手元にあるときだけ使う。表は流れの順に並べた。「直下の scripts/」はリポジトリの直下の `scripts/` にあるもの。コマンドはすべてリポジトリのルートから実行する。

| フォルダ | スクリプト | 内容 | 入力 -> 出力 |
|---|---|---|---|
| setup | `setup_edge.sh` | EDGE と Jukebox を `~/external` に入れ、専用の venv を作る | なし -> `~/external/EDGE`、`~/.venvs/edge` |
| 直下の scripts/ | `fetch_basketball.sh` | ボール上で立つ b11 のチェックポイントを取得して SHA256 を確かめる | Hugging Face -> `logs/rsl_rl/basketball/b11_6999/checkpoint.pt` |
| 直下の scripts/ | `fetch_pollen_policies.sh` | Pollen の roller.onnx と芸の policy（roulade、alpha_sitstand、velstand）を commit 665ae4eb で取得して SHA256 を確かめる | Hugging Face -> `logs/tricks/pollen-microduck-policies/*.onnx` |
| generate | `synth_music.py` | 学習用の拍だけの曲を numpy で合成する | シード -> `data/music/synthetic/synthetic_NN.wav` |
| generate | `compute_beats.py` | librosa で拍の時刻と音楽の特徴を求める | wav -> `data/beats/{songs,training}/*.json` |
| generate | `edge_generate.py` | EDGE で曲から人のダンス（SMPL）を生成する | wav -> pkl（30 fps） |
| generate | `generate_motion_edge.sh` | `edge_generate.py` を EDGE の venv で呼ぶ。1曲でも `--jobs` の一括でも使える | wav -> `data/motions/songs/<曲名>.pkl` |
| generate | `generate_library_edge.sh` | 学習用33曲のダンスを作り、浮いた動きは作り直す | `data/music/synthetic/*.wav` -> `data/motions/training/*.pkl` |
| generate | `schedule_tricks.py` | 曲の盛り上がりから芸（前転、おすわり）を入れる時刻を選ぶ | 拍の json -> `data/tricks/<曲名>.json` |
| retarget | `retarget_motion.py` | 人の動きを13次元のロボットコマンドに変換する（groove と phrase gate を含む） | pkl（と拍の json） -> npz（50 Hz） |
| retarget | `retarget_all.sh` | 全 pkl を変換し、学習用はライブラリにまとめる | `data/motions/*` -> `data/commands/*`、`data/library.npz` |
| retarget | `build_library.py` | 複数の npz を1つの学習用ライブラリにつなぐ | `data/commands/training/*.npz` -> `data/library.npz` |
| retarget | `library_stats.py` | コマンドの次元ごとの統計を出し、分布を比べる | npz -> 標準出力 |
| retarget | `fit_posture_synergy.py` | dance_v3 の姿勢 synergy を足裏を床につけた IK で当てはめる | ロボットモデル -> 係数の表（標準出力） |
| 直下の scripts/ | `onnx_to_checkpoint.py` | Pollen の roller.onnx を rsl_rl の actor の重みに戻し、出力が ONNX と一致するかを確かめる | roller.onnx -> `logs/rsl_rl/roller_dance/roller_onnx_init/actor.pt` |
| 直下の scripts/ | `finetune_roller_dance.py` | roller.onnx の actor をローラーのダンスのタスク `Mjlab-RollerDance-MicroDuck` で追加学習する（critic のウォームアップ、再開時の rewarm） | actor.pt か run の checkpoint -> `logs/rsl_rl/roller_dance/<run>/model_*.pt` |
| 直下の scripts/ | `finetune_basketball.py` | b11 を ball dance のタスク `Mjlab-BallDance-MicroDuck` で継続学習する（`--task`）。上流の同名のスクリプトから派生 | b11 の checkpoint.pt -> `logs/rsl_rl/<タスクの experiment_name>/<run>/` |
| 直下の scripts/ | `check_ball_dance_step0.py`、`measure_ball_stance.py` | ball dance の継続学習の最初の1歩が b11 と同じかの確認と、ボール上の立ち姿勢の計測 | b11 の checkpoint.pt -> 標準出力 |
| evaluate | `play_dance.py` | 学習した policy でコマンドを再生し、録画か追従の計測をする | チェックポイント、npz -> mp4、json、状態の npz |
| evaluate | `gate_check.py` | dance_v3 のチェックポイントが基準を通るかを調べる | チェックポイント -> 合否と数値 |
| evaluate | `score_motion.py` | 人の動きがロボットに見える動きをどれだけ持つかを点数にする | pkl と wav の対 -> csv |
| evaluate | `visible_motion_table.py` | 状態の npz から見える動きの表を作る | `play_dance.py --dump-states` の npz -> 表、json |
| evaluate | `fidelity_eval.py` | 人の動きと実現した動きの対応（チャンネルごとの相関、移動量、向きの誤差、曲間の違い）を出す。fid001 の表の測定 | `play_dance.py --dump-states` の npz、人の pkl -> `out/fid001/fidelity_*.json` |
| evaluate | `travel_ratio.py` | 人とロボットの移動量を身長単位で比べ、比を出す（vel_gain を選ぶ基準） | 状態の npz、人の pkl -> 標準出力 |
| evaluate | `sweep_vel_gain.sh` | vel_gain ごとに再変換して、3曲を録画モードで再生し、状態を保存する | `retarget_motion.py`、チェックポイント -> `out/fid001/sim/*_states.npz` |
| evaluate | `fall_eval.sh` | 現在のコマンドで3曲の転倒率を測る（seed 0、16 env） | チェックポイント、`data/commands/songs/*.npz` -> `out/fid001/eval/*.json` |
| evaluate | `rough_eval.sh` | 平らな床と凸凹の地面（タイルの台の外）で3曲の転倒率を測る（seed 0、16 env） | チェックポイント -> `out/rough/eval/<name>/*.json` |
| evaluate | `rough_record.sh` | 平らな床、段差、格子の上で3曲の状態を記録し（`fidelity_eval.py` 用）、段差の上の15秒を録画する | チェックポイント -> `out/rough/sim/*_states.npz`、`out/rough/*.mp4` |
| evaluate | `rough_metrics.py` | `play_dance.py --eval --dump-eval` の記録から、環境ごとに人との向きと前後の移動の相関を出し、転倒と頭の RMSE を添える（rough002 以降の凸凹の表） | npz（と同じ名前の json） -> json |
| evaluate | `terrain_figure.py` | 凸凹の地形の4種類を、ロボットと同じ縮尺の断面つきで1枚の図にする（render は EGL、compose は日本語フォント） | 学習用の地形の cfg -> `dance/docs/figures/terrain_types.png` |
| evaluate | `roller_dance_eval.py` | ローラーでダンスのコマンドを再生し、転倒、追従、移動量を測るか、録画する | チェックポイントか actor.pt、npz -> json、mp4 |
| evaluate | `chain_trick.py` | 予定表どおりに踊りから芸へ切り替え、起き上がりを確かめて踊りに戻し、成功数を数える | チェックポイント、Pollen の ONNX、`data/tricks/*.json` -> json、mp4 |
| video | `render_human_3d.py` | 人の動きを MuJoCo の3Dマネキンで描く | pkl -> mp4 |
| video | `render_human.py` | 人の動きを2Dの棒人間で描く。`render_human_3d.py` が関節の親子関係と pkl の読み込みを使うので残している | pkl -> mp4 |
| video | `pick_segment.py` | コマンドから最も動きのある区間を選ぶ | npz -> 開始時刻（json） |
| video | `compose_video.py` | ffmpeg で人とロボットの動画を並べ、曲と題字を重ねる | mp4、wav -> mp4 |
| video | `make_final_video.py` | 提出用の動画を組み立てる（3曲、比較、凸凹、ローラーの区間） | `out/rec_v4/*`、`out/eval_v4/*`、凸凹とローラーのクリップ -> 動画 |
| video | `build_report_pdf.sh` | `dance/docs/report.md` から説明資料の PDF を作る | report.md -> `out/report/*.pdf` |

以前の実験では FlowerDance でダンスを生成した。FlowerDance にはライセンスが無いので、FlowerDance を呼ぶスクリプトと FlowerDance で作ったファイルはこの公開リポジトリに入れていない。`compute_beats.py` と `score_motion.py` は EDGE の venv（`setup/setup_edge.sh`）で動く。

兄弟のスクリプトを読み込むものは同じフォルダの中にある（`gate_check.py` から `play_dance.py`、`make_final_video.py` から `compose_video.py` と `pick_segment.py`、`render_human_3d.py` から `render_human.py`、`compute_beats.py` から `synth_music.py`）。

fid001 の測定の順番は、`sweep_vel_gain.sh vg0.3 ...`、`fidelity_eval.py base vg0.3 ...`、`travel_ratio.py base vg0.3 ...`、`fall_eval.sh`。`fidelity_eval.py` と `travel_ratio.py` は `uv sync` で作った venv で動かす。GPU を使う2つのシェルは、GPU のメモリ使用が `GPU_MAX_MB`（既定 7000）未満になるまで待つ。
