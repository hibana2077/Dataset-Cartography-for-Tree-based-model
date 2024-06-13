<!--
 * @Author: hibana2077 hibana2077@gmail.com
 * @Date: 2024-06-12 18:09:49
 * @LastEditors: hibana2077 hibana2077@gmail.com
 * @LastEditTime: 2024-06-12 22:28:33
 * @FilePath: \Dataset-Cartography-for-Tree-based-model\docs\main.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# 實驗規劃

## Index

- [1. 實驗目的](#1-實驗目的)
- [2. 實驗計畫](#2-實驗計畫)

## 1. 實驗目的

確認使用 dataset cartography 的分割資料方式，再把這些資料用於訓練獨立模型，查看是否能夠有效提升模型準確度

## 2. 實驗計畫

1. 找多批資料集
2. 使用 cartography 的方式，對資料集進行建模
3. 統計檢定，確認是否有提升模型準確度 (同質性檢定, t-test, bayesian)

### 實驗數據收集

為了完成統計檢定，需要收集以下數據：

- 一開始的 dataset cartography
- 一開始的 評估指標 test and train (cm, f1, acc, recall, precision)
- stack model 的 評估指標 test and train (cm, f1, acc, recall, precision)