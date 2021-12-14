# UnetModelScript
TF TutorialのPix2PixのGeneratorを参考にした

## メモ
- 1126
    - 全く同じデータを用いても学習序盤に非常に大きな差
    - Pix2Pixの実装の生成器を参考にする

## メモ1214
- Augmentをtf.data.Datasetに対して.mapでnum_parallelを有効にして並列に計算すると同じシードをinpとtarに用いて適用しても、
どうやら異なる変換が施される。

シードが同じでも並列処理しているためことなる乱数列が適用される??????

    
