# Standard LSTM Modal
标准LSTM模型的代码实现（python版本）


数据集：
训练语料（已经打好标签）：pos.xls  neg.xls  sum.xls

迭代次数建议改成5次或者更多：
```
model.fit(xa, ya, batch_size=16, nb_epoch=5,validation_data=(xa, ya))  # 第89行
```