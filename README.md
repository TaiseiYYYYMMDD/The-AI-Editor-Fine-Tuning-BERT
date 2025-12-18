# The-AI-Editor-Fine-Tuning-BERT
Build a custom AI model that reads Japanese news titles/articles and  automatically classifies them into specific categories (e.g., "Sports," "IT,"  "Entertainment"). 

# The AI Editor (BERT Fine-Tuning)

日本語テキストを入力すると、学習済みBERTモデルを用いてカテゴリ分類を行うWebアプリです。  
Flask を用いたシンプルなアプリ構成になっています。

---

## フォルダ構成


The-AI-Editor-Fine-Tuning-BERT/
├─ app/
│ ├─ app.py # Flask アプリのエントリーポイント
│ ├─ inference.py # モデル読み込み・推論処理
│ ├─ templates/
│ │ └─ index.html # 画面UI
│ └─ static/
│ └─ style.css # スタイル
│
├─ model_build/
│ └─ ai_editor_model/ # ★学習済みモデルを配置する場所（Git管理外）
│
├─ requirements.txt
├─ README.md
└─ .gitignore


---

## 重要：モデルについて

`model_build/ai_editor_model/` フォルダには **学習済みモデル一式** を配置してください。

⚠ このフォルダは **GitHubには含まれていません**（容量制限回避のため）。  
モデルは別途共有されたものを使用してください。

### 必要なファイル例
model_build/ai_editor_model/
├─ config.json
├─ model.safetensors
├─ tokenizer_config.json
├─ vocab.txt
└─ special_tokens_map.json