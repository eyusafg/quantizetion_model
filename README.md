# quantizetion_model
# 原始模型结构：
Conv -> ReLU -> ReduceMax -> Sigmoid

排除后的量化情况：
[量化]Conv -> [量化]ReLU -> [保持浮点]ReduceMax -> [保持浮点]Sigmoid
因为量化的分割模型， 但是reducemax对数值精度很敏感， 后处理对精度要求也高， 所以排除reducemax节点的量化
