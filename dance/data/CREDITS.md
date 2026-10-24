# Music credits

All five songs come from OpenTracks (formerly DOVA-SYNDROME; the site was
renamed on 2026-09-15 and dova-s.jp now redirects to opentracks.com).
Files are not in git. `dance/data/music/opentracks/<song>.wav` is the first 60 s of the
downloaded mp3, resampled to 44.1 kHz stereo with a 1.5 s fade out at the end
(the license allows format conversion, cutting and fades).

Checked on 2026-10-19.

## Songs

| song (file stem) | use | title | composer | track page | BPM | composer conditions | credit required |
|---|---|---|---|---|---|---|---|
| swag | video | Swag | Phalene | https://opentracks.com/bgm/detail/9564 (track 1) | about 92 (librosa) | https://opentracks.com/creator/detail/260 : 「R-18コンテンツでの使用は禁止しております。」 | No |
| funky_magic | video | Funky Magic | FLASH☆BEAT | https://opentracks.com/bgm/detail/10397 (track 1) | 120 (track page) | https://opentracks.com/creator/detail/261 : site license only (サイト準拠). 「楽曲をご利用の際に【FLASH☆BEAT】と表記していただけるととても嬉しいです。（強制ではありません）」 | No (requested, optional) |
| fallin_for_you | video | Fallin For You (Track 2, instrumental) | Flehmann | https://opentracks.com/bgm/detail/14973 (track 2) | about 118 (librosa) | https://opentracks.com/creator/detail/410 : site license only (サイト準拠) | No |
| redwing_beats | spare | Redwing Beats | Anonyment | https://opentracks.com/bgm/detail/13106 (track 1) | 110 (track page) | https://opentracks.com/creator/detail/23 : site license only (サイト準拠). 「ご任意ですが、コンテンツにご使用の際はクレジット表記のご協力をお願いしております。( Music : Anonyment 等)」 | No (requested, optional) |
| pastel_house | spare | パステルハウス | かずち | https://opentracks.com/bgm/detail/1021 (track 1) | about 118 (librosa) | https://opentracks.com/creator/detail/50 : 「成人向けの表現がある作品でも使用していただいて大丈夫です。」 (a relaxation, no extra restriction) | No |

Suggested YouTube description line (credit is optional, but it helps when
answering a mistaken claim):

```
Music: "Swag" by Phalene / "Funky Magic" by FLASH☆BEAT / "Fallin For You" by Flehmann (OpenTracks, formerly DOVA-SYNDROME, https://opentracks.com/)
```

## Library music

The music for the training library (`dance/data/music/synthetic/`, not in git) is
synthesized by `dance/scripts/generate/synth_music.py` from numpy oscillators and noise.
It has no third-party rights. The OpenTracks songs above are not used for the
library, because their license forbids AI training.

## Dance generation models

Since 2026-10-20 (Task 14) the dances in `dance/data/motions/songs/` and
`dance/data/motions/training/` come from EDGE. Only the pretrained models are run
(inference). Nothing is trained on the songs or on the AIST++ data here.

- Dance generation: EDGE, "EDGE: Editable Dance Generation From Music",
  J. Tseng, R. Castellon, C. K. Liu, CVPR 2023.
  Code https://github.com/Stanford-TML/EDGE (MIT license), used at commit
  17c3428669ed6733edd9d8c66f7dc62060b8e46d with the released checkpoint.
- Training data of the EDGE checkpoint: AIST++, "AI Choreographer: Music
  Conditioned 3D Dance Generation with AIST++", R. Li, S. Yang, D. A. Ross,
  A. Kanazawa, ICCV 2021. Annotations under CC BY 4.0
  (https://google.github.io/aistplusplus_dataset/). The AIST++ motions come
  from the AIST Dance Video Database (S. Tsuchida et al., ISMIR 2019). Its
  terms of use (https://aistdancedb.ongaaccel.jp/terms_of_use/, checked on
  2026-10-24) say: "AIST Dance DB may not be used for any purpose other than
  academic research." / "Use for commercial purposes is not permitted without
  prior written consent from AIST." / "Unauthorized redistribution of any
  content of the database is prohibited." / "it is necessary for you to clearly
  indicate the name of the database". We do not use or redistribute the
  database itself. The dances here come from a model trained on data derived
  from it, so we keep them non-commercial and name the database in the credits.
- Music features: OpenAI Jukebox, "Jukebox: A Generative Model for Music",
  P. Dhariwal, H. Jun, C. Payne, J. W. Kim, A. Radford, I. Sutskever, 2020.
  Code and weights under the OpenAI Noncommercial Use License
  (https://github.com/openai/jukebox/blob/master/LICENSE), which requires
  credit. Used only as a feature extractor (top prior layer 66, through
  jukemirlib, MIT, https://github.com/rodrigo-castellon/jukemirlib), no audio is
  generated. This is a non-commercial university assignment.
  Video credit line: "Dance motion generated with EDGE (Tseng et al., CVPR 2023)
  trained on AIST++; music features from OpenAI Jukebox."
- Earlier experiments (until 2026-10-20): FlowerDance
  (https://github.com/XulongT/FlowerDance), trained on FineDance
  (non-commercial research license). FlowerDance has no license, so the
  public repository contains neither the scripts that call it nor its clips
  or the files made from them.

## Trick policies

Since 2026-10-23 (Task 23) the dance can hand over to a trick policy at a musical
climax (`dance/scripts/evaluate/chain_trick.py`). The trick policies are not ours.
They come from `pollen-robotics/microduck-policies`, the policy set a microduck
ships with, used unmodified as ONNX files.

- Source: https://huggingface.co/pollen-robotics/microduck-policies at commit
  665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448.
- License: Apache-2.0. Checked on 2026-10-23 from the Hugging Face API
  (`cardData.license` = `apache-2.0`) and from the front matter of the repository
  README at that commit (`license: apache-2.0`). The repository has no separate
  LICENSE file.
- Files used (SHA256 as downloaded, equal to the Hugging Face LFS record):

| file | use | SHA256 |
|---|---|---|
| `roulade.onnx` | front roll (roulade) | 3d60da08fc13f29c1b57f41977aa898132c0d60042100149d8e775affcbca32b |
| `alpha_sitstand.onnx` | sit and stand | a6f102d9fd655782ba85bfffe18175c34621da719cffcdd673be7982a06fc254 |
| `velstand.onnx` | standing policy that holds between a roulade and the dance | 2fc7f8ee6acd7d0d3f66a99927b5d5a56589aeed02ec714de329b252d5ccf492 |

- The files are not in git (`logs/tricks/pollen-microduck-policies/`). Our own
  parts are the scheduler, the hand-over state machine and the evaluation
  (`src/microduck_dance/tricks.py`). The schedules in `dance/data/tricks/`
  are computed from the beats of the songs above and contain no third-party content.
- Video credit line: "Trick policies (roulade, sitstand, velstand) from
  pollen-robotics/microduck-policies, Apache-2.0."

## Roller policy (modified)

Since 2026-10-23 (Task 27) the roller skate dance (`Mjlab-RollerDance-MicroDuck`)
starts from Pollen's roller skating policy and **modifies its weights** by
further training. This is a modification in the sense of Apache-2.0 section 4(b).

- Source: `roller.onnx` from https://huggingface.co/pollen-robotics/microduck-policies
  at commit 665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448
  (https://huggingface.co/pollen-robotics/microduck-policies/blob/665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448/roller.onnx).
- SHA256 as downloaded (equal to the Hugging Face LFS record):
  cf05651d2708a2f9364212e86b866c97a70ace8131c492500105e8f28bf99afd.
- License: Apache-2.0, the same repository level evidence as the trick policies above.
- What we changed. `scripts/onnx_to_checkpoint.py` copies the ONNX weights and
  observation normalizer into an rsl_rl actor. The normalizer columns of the
  head and body command slots are reseeded with library statistics and their
  input weights set to 0. The actor is then trained further with PPO on our
  roller dance task (`scripts/finetune_roller_dance.py`). The resulting weights
  are ours plus theirs and are not the original file.
- Neither the original file nor the modified weights are in git
  (`logs/rsl_rl/roller_dance/`). If the modified policy is redistributed, it must
  carry the Apache-2.0 text, this notice of modification and the source above.
- Video credit line: "Roller skating policy initialised from
  pollen-robotics/microduck-policies roller.onnx (Apache-2.0), modified by further training."

## Robot model and meshes

- The robot model, the task framework and the walking task come from
  microduck-playground (https://github.com/Vottivott/microduck-playground,
  software under Apache-2.0, `LICENSE` and `NOTICE` in its repository root),
  included here as the git submodule `third_party/microduck-playground` at commit f010ef1,
  which derives from pollen-robotics/microduck_rl
  (https://github.com/pollen-robotics/microduck_rl, Apache-2.0).
- The 3D meshes of Microduck
  (`third_party/microduck-playground/src/mjlab_microduck/robot/microduck/assets/*.stl`)
  are by Pollen Robotics under CC BY-NC-SA 4.0
  (https://creativecommons.org/licenses/by-nc-sa/4.0/). Source and terms:
  `LICENSE-MODELS.md` of https://github.com/pollen-robotics/microduck_rl
  (branch develop, checked on 2026-10-24) and `LICENSE-HARDWARE` in the submodule. Its
  attribution rules: credit Pollen Robotics with the link above, link the
  license, say whether the files were modified, and license an adaptation under
  CC BY-NC-SA 4.0.
- Modified or not: no mesh file is changed. They are only in the submodule,
  which is the upstream commit f010ef1 without changes.
  The video shows renders of the unmodified meshes.
- Video credit line (end card): "ロボットのモデルとメッシュ Microduck, Pollen
  Robotics (CC BY-NC-SA 4.0、メッシュは改変なし)" followed by the license and
  source URLs. Put the same two URLs in the YouTube description.

## Ball policy b11

The ball dance (`Mjlab-BallDance-MicroDuck`, ball003, not in the video) starts
from the released basketball policy b11 and changes its weights by further
training.

- Source: https://huggingface.co/HannesVonEssen/microduck-basketball,
  `checkpoint.pt` (iteration 6999), repository commit
  6e61a733aeff338bdd352148819bc99cf9a51491, downloaded by
  `scripts/fetch_basketball.sh`. SHA256
  57a322eff092cb71e7cba831791232f0cc4fd45ede0441cad6d87813b0033e41, equal to the
  repository's `SHA256SUMS`.
- License: Apache-2.0. Checked on 2026-10-24 from the primary source. The
  Hugging Face API gives `"private": false`, `"gated": false` and
  `cardData.license` = `apache-2.0`. The README front matter says
  `license: apache-2.0`. The repository has a `LICENSE` file with the Apache
  License 2.0 text and a `NOTICE` file ("MicroDuck Playground / Copyright 2026
  Hannes von Essen / Vottivott ... Software: Apache License 2.0.").
- Neither the original file nor the modified weights are in git
  (`logs/rsl_rl/basketball/`, `logs/rsl_rl/ball_dance/`). If the modified policy
  is redistributed, it must carry the Apache-2.0 text, the `NOTICE` file, a
  notice of modification and the source above.

## Simulation and training software

Installed by `uv sync` from `pyproject.toml`, not redistributed here. Licenses
from the installed package metadata (checked on 2026-10-24): mjlab 1.3.0
Apache-2.0, MuJoCo 3.10.0 Apache-2.0, MuJoCo Warp 3.8.1 Apache-2.0, NVIDIA Warp
1.12.0 Apache-2.0, rsl_rl (rsl-rl-lib 5.0.1) BSD-3-Clause.

## Course opening and ending clips

The final video starts and ends with the course's opening and ending clips
(`out/course_assets/op.mp4`, `ed.mp4`, 3 s each, no audio). They were handed out
by the course for the submission video. They are not in this repository and
must not be committed. The assignment README of the course says nothing about
how these clips may be used. `make_final_video.py` skips both sections with a
warning when the files are missing.

## License evidence

Sound license (音源利用ライセンス): https://opentracks.com/help/articles/license/

YouTube posting is allowed as background music:

> ライセンスの範囲 1. 当ライセンスおよび作曲・制作者の制定した利用条件による限り、営利（商用）・非営利、また、個人・法人を問わず、ユーザーは、あらゆる制作物に対して音源を以下の態様により利用することができます。（a）背景音楽としての利用

> 背景音楽としての利用例 ... 学校等の課題のための利用 / YouTube、TikTok、Twitch、IRIAM、Podcast、その他のオンライン映像・コンテンツ配信での利用

No credit or fee is required:

> 前提条件 1. 音源の利用に際し、著作権使用料その他の利用料を請求することはありません。 2. ユーザーによる音源のご利用にあたり、著作権表示・提供等の表示は不要です。

Composer conditions cannot override the no-fee, no-credit and prohibited-use clauses:

> ※但し、この前提条件「1」「2」および後述の「禁止事項」については当ライセンスが優先します。

Editing is allowed:

> 2. ユーザーは、音源の加工（編集・エフェクト・フェードイン/アウト等）を行うことができます。

Prohibited uses that matter for this project:

> 5. 音源を作品等に使用せず（付加価値を加えず）、「作業用BGM」「ヒーリングミュージック」「音楽紹介を主なコンテンツとするWEBラジオ」等、音源を主なコンテンツとする動画等を公開すること

> 11. 音源をAIのトレーニングに使用すること

> 14. 音源および当該音源に係る楽曲並びに音楽作品その他の制作物（本ライセンスの範囲内であるか範囲外であるかを問いません）を、音楽としてUGCサービスその他のプラットフォームにおけるフィンガープリント（YouTubeにおいてはContent IDをいうがこれに限られません）に登録すること

How we comply. The robot video is the main content and the song is its
background. The songs are only fed to pretrained models (EDGE with Jukebox
features since 2026-10-20, FlowerDance before) at inference time. Nothing is trained on them. We never register the video or
the songs with Content ID.

YouTube help page: https://opentracks.com/help/articles/youtube/

> 当サイトの音源を使用して制作した動画のYouTubeでの公開・配信について、ライセンスの範囲内であれば問題なく無料で利用いただけます。

> 収益化・商用利用 法人・個人・営利・非営利を問わず、利用料や著作権表示は原則不要です。

Content ID note. OpenTracks tracks are managed on YouTube by TRACKS Inc.
(「当サイトでは、音源のYouTube上における著作権管理を株式会社TRACKSが行っています。」).
A claim from TRACKS Inc. is expected only for videos that break the license.
The same page reports that some compliant videos got a temporary claim from
"TRACKS Inc.（Publishing）", which TRACKS resolves, and that overseas groups
sometimes file mistaken publishing claims. If a claim appears, follow that
page.

## 利用上の注意

- ライセンスの禁止事項11「音源をAIのトレーニングに使用すること」により、これらの曲をAIの学習に使ってはいけない。本プロジェクトでの使い方は推論だけである（学習済みの生成モデル EDGE、以前は FlowerDance に曲を入力してモーションを得る）。したがって、これらの曲や曲から作った特徴量を、強化学習の学習データ（コマンド分布の作成、報酬の参照、policy の学習用モーション）を作るために使ってはいけない。
- 禁止事項14により、動画や曲を Content ID に登録しない。

## YouTube 投稿前の手順

1. **限定公開で先にアップロードする。** 公開する前に、YouTube Studio で該当動画の著作権の表示（申し立ての有無）を確認する。申し立てがなければ公開に切り替える。
2. **申し立てが付いた場合に起こること。** 記録された出典にある範囲だけを書く。
   - YouTube ヘルプ「Learn about copyright claims」(https://support.google.com/youtube/answer/6013276) によると、権利者の Content ID の設定次第で次のいずれかになる。"Block a video from being viewed" / "Monetize the video by running ads on it, sometimes sharing revenue with the uploader" / "Track the video's viewership statistics"。同ページには "You will not receive a copyright strike for videos that contain active claims." とある。ミュートについてはこのページに記載がないので、ここでは扱わない。
   - OpenTracks の YouTube ページ (https://opentracks.com/help/articles/youtube/) は、海外団体による誤った権利主張で起こりうる問題として「動画に広告が表示されない」「動画の広告収益の一部が没収される」「動画の説明に楽曲のクレジットが表示される」を挙げている。
3. **異議申し立ての手順。**
   - まず申し立て元を確認し、OpenTracks の手順 (https://opentracks.com/help/articles/youtube/ の「著作権侵害の申し立てを受けた場合」) に従う。「TRACKS Inc.（Publishing）」の場合は一時的な不具合とされ、「YouTube上で異議の申立をしていただければ、対象動画に対し速やかに対応が行われます。」とある。TRACKS の案内ページは https://www.tracks.co.jp/dova_youtube.html。海外団体の場合は「申立を受けたご本人様よりYouTube上にて、異議申し立てを行っていただくか、またはYouTubeのサポートへお問合せいただく必要がございます。」とあり、YouTube ヘルプ https://support.google.com/youtube/answer/2807684?hl=ja が案内されている。どれにも当たらない場合は、対象動画 URL、申立元、権利所有者を添えて OpenTracks のお問合せフォームに連絡する。
   - 手元に用意するもの: 各曲のページ URL（上の表）、ライセンスのページ https://opentracks.com/help/articles/license/、および次の条文。
     - 「ユーザーは、あらゆる制作物に対して音源を以下の態様により利用することができます。（a）背景音楽としての利用」
     - 「YouTube、TikTok、Twitch、IRIAM、Podcast、その他のオンライン映像・コンテンツ配信での利用」
     - 「2. ユーザーによる音源のご利用にあたり、著作権表示・提供等の表示は不要です。」
     - 「当サイトの音源を使用して制作した動画のYouTubeでの公開・配信について、ライセンスの範囲内であれば問題なく無料で利用いただけます。」
4. **概要欄のクレジット（任意）。** 必須ではないが、誤った申し立てへの対応に役立つので入れる。

   ```
   Music: "Swag" by Phalene / "Funky Magic" by FLASH☆BEAT / "Fallin For You" by Flehmann (OpenTracks, formerly DOVA-SYNDROME, https://opentracks.com/)
   ```

   予備曲を使う場合は "Redwing Beats" by Anonyment、"パステルハウス" by かずち を加える。
5. **申し立ての履歴。** OpenTracks は曲ごとの申し立て履歴を公開していない。5曲の曲ページと作曲者ページにも申し立てについての記載はなかった（2026-10-19 確認）。したがって5曲とも履歴は不明である。

## Sites that were not used

- YouTube Audio Library: needs a signed-in YouTube Studio session
  (https://support.google.com/youtube/answer/3376882, "Sign in to YouTube Studio" then "Audio library"). Skipped because we do not log in or create accounts.
- Pixabay Music: the music pages and https://pixabay.com/service/license-summary/
  return a Cloudflare challenge to curl (HTTP 403, "Just a moment..."). Skipped as a CAPTCHA gate. The license summary (read through a web fetcher) says "Use Content without having to attribute the author". Its Content ID warning was not checked here.
