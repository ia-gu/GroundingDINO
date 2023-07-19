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
→bbox.jsonにGTのBBが，vtec/*/bounding_boxes/下に画像が保存される

性能検証
```
evaluation.py
```
→result.jsonに各画像のIoUが，result/下に予測画像が保存される

※predict()を別のモデルに変えれば他も検証可能
