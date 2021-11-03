# Self-driving car simulator

## Teacher's requirement
1. GUI 介面:
甲、讀取軌道(如”case01.txt”)並根據軌道座標點繪製軌道及自走車
乙、以動畫顯示自走車每一時刻的位置及行走方向
丙、顯示左、前、右 3 個測距 sensor 測得的距離
2. 紀錄自走車行駛路徑，紀錄格式下面會說明
3. 讀取行進路徑記錄檔讓自走車根據紀錄檔中的路徑行走碰撞偵測，自走
車碰到軌道及終點須能自動停止
4. 將左、前、右 3 個測距 sensor 測得的距離當作模型輸入，利用 MLP 或
RBFN 訓練出一個模型可以使車子順利抵達終點。
附檔訓練資料集有兩種格式的移動紀錄：train4D.txt、train6D.txt
train4D.txt 格式:前方距離、右方距離、左方距離、方向盤得出角度(右轉
為正)
train6D.txt 格式:X 座標、Y 座標、前方距離、右方距離、左方距離、方
向盤得出角度(右轉為正)
## Features
* MLP
    - \[finish\] 能夠設定層數和每層有多少神經元 
    - \[finish\] activation function 可替換 
    - 能夠儲存模型 
* GUI
    

* CAR
    - record car coordinate



