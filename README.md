# fx-briefing
夜のブリーフィング用データパックを毎日自動作成する（発注はしない）。
- GitHub Actions が平日18:41/19:11(JST)に `datapack.py` を実行 → `docs/datapack.json` を更新
- Claude の定時タスクが19:25に読み、ブリーフィングを書き、`verify.py` で数字を照合してから配信
- 判定ルールは「夜のブリーフィング_仕様.md」（claude.ai FX攻略プロジェクト）
- 口座残高は `config.json` の `equity_jpy` を手で更新（ロット計算に使用）
