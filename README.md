# GroundingDINO
GroundingDINOの検証用リポジトリ

### 事前準備

```
pip install -r requirements.txt
pip install -q roboflow
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```
→demo.ipynbでデモ

### 実験
セグメンテーションマスクからバウンディングボックス（BB）を作成
```
python seg2det.py
```
→evaluation.pyで性能検証

predict()を別のモデルに変えれば他も検証可能（PROのためにBBの形は揃える）
