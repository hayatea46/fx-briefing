# fx-briefing
裁量トレード用の情報を毎日自動作成する（発注はしない）。
- 朝07:23：前日までの確定足で各ペアの方向（ロング/ショート/方向なし）と無効化ライン → `docs/morning.json`、`docs/state/日付.json`
- 夕方18:53：昼の1時間足で朝の方向が「維持/弱まった/無効」か確認 → `docs/evening.json`
- 夜20:03〜翌01:03 毎時：ロング推奨/ショート推奨/見送りと価格帯をスマホへ通知（ntfy）→ `docs/hourly.json`
- 夜の推奨は「朝の方向と同じ向き」か「見送り」のみ（途中で逆向きにはしない）
- 朝と夕方はClaudeが `verify.py` で数字を照合したうえで解説を配信
- 口座残高は `config.json` の `equity_jpy` を手で更新（ロット計算用）
- Secrets：`TWELVEDATA_API_KEY`（ゴールド現物）、`NTFY_TOPIC`（通知先）
