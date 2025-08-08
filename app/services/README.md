
# 面部臟腑識別分析系統

## 輸出資料夾說明

### 1. output_organs/
包含所有裁切的臟腑區域小圖片
- 格式：原檔名_臟腑名稱.jpg
- 例如：photo001_heart.jpg, photo001_liver.jpg

### 2. marked_images/
包含在原圖上標記臟腑位置的圖片
- 每個臟腑區域用不同顏色的矩形框和中心點標示
- 純視覺標記，無文字標籤
- 格式：marked_原檔名.jpg

### 3. processed/
包含格狀分析結果
- original_*.jpg: 調整尺寸後的原圖
- grid_*.jpg: 100x100網格分析圖
- dark_blocks_*.jpg: 異常區域處理圖

## 臟腑對應顏色
- 腦(brain): 紫色
- 肺(lung): 青色  
- 心(heart): 紅色
- 肝(liver): 綠色
- 脾(spleen): 橙色
- 腎(kidney): 藍色
- 胃(stomach): 黃色
- 大腸(long): 深紫色
- 膀胱(bladder): 深橙色

## 使用方法
1. 將待分析的人臉照片放入 images/ 資料夾
2. 執行程式
3. 查看三個輸出資料夾的結果

注意：此系統僅供研究參考，不可用於醫療診斷。
