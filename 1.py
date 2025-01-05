import os
import onnx
import onnxruntime as ort
import numpy as np
import cv2
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_static, QuantType, CalibrationMethod
from onnxruntime import InferenceSession, get_available_providers

onnx_model_path = 'thor_segm0103.onnx'
model_quant_static_path = 'quant_model_99_0103.onnx'

img_dir = r'ims'
datas = [np.expand_dims(cv2.resize(cv2.imread(os.path.join(img_dir, file)), (640,640))[:, :,::-1].transpose(2, 0, 1), axis=0).astype(np.float32) for file in os.listdir(img_dir)]


# Create a calibration data reader for the images
def batch_reader(datas, batch_size):
    _datas = []
    length = len(datas)
    for i, data in enumerate(datas):
        if batch_size == 1:
            print('data shape is: ', data.shape)
            yield {'input_image': data}
        elif (i+1) % batch_size == 0:
            _datas.append(data)
            yield {'input_image': np.concatenate(_datas, 0)}
            _datas = []
        elif i < length - 1:
            _datas.append(data)
        else:
            _datas.append(data)
            yield {'input_image': np.concatenate(_datas, 0)}


class DataReader(CalibrationDataReader):
    def __init__(self, datas, batch_size):
        self.datas = batch_reader(datas, batch_size)
    
    def get_next(self):
        return next(self.datas, None)


# 加载模型，查看输入要求
onnx_model = onnx.load(onnx_model_path)
print("Model Input Info:")
for input in onnx_model.graph.input:
    print(input.name, [dim.dim_value for dim in input.type.tensor_type.shape.dim])

for node in onnx_model.graph.node:
    if node.op_type == 'ReduceMax':
        print(f"Node name: {node.name}")
        print(f"Axes: {node.attribute[0].ints}")

# 加载模型并创建推理会话
sess = ort.InferenceSession(onnx_model_path)
input_name = sess.get_inputs()[0].name

# 获取 ReduceMax 节点的输出名称
for node in onnx_model.graph.node:
    if node.op_type == 'ReduceMax':
        reduce_max_output_name = node.output[0]
        break



import onnx

# 加载模型
onnx_model = onnx.load('thor_segm0103.onnx')

# 遍历所有节点并打印 ReduceMax 节点的输出名称
# for node in onnx_model.graph.node:
#     if node.op_type == 'ReduceMax':
#         print(f"Node name: {node.name}")
#         for idx, output in enumerate(node.output):
#             print(f"Output {idx}: {output}")

reduce_max_output_names = []
for node in onnx_model.graph.node:
    if node.op_type == 'ReduceMax':
        for output_name in node.output:
            reduce_max_output_names.append(output_name)

print("ReduceMax output names:", reduce_max_output_names)

# 找到 Resize 节点
resize_node = []
for node in onnx_model.graph.node:
    if node.op_type == 'Resize':
        resize_node.append(node)
        

if resize_node:
    for node in resize_node:
        print(f"Node name: {node.name}, Op type: {node.op_type}")
        for input_name in node.input:
            print(f"Input name: {input_name}")
        for output_name in node.output:
            print(f"Output name: {output_name}")

# from onnx import helper, checker, ModelProto, GraphProto, NodeProto, ValueInfoProto, TensorProto

# # 加载原始模型
# model = onnx.load(onnx_model_path)

# # 找到 ReduceMax 节点并将其输出添加为模型输出
# reduce_max_outputs = []
# for node in model.graph.node:
#     if node.op_type == 'ReduceMax':
#         for output_name in node.output:
#             # 创建 ValueInfoProto 对象用于新的输出
#             value_info = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, None)
#             reduce_max_outputs.append(value_info)

# # 更新模型的输出列表
# model.graph.output.extend(reduce_max_outputs)

# # 保存修改后的模型
# new_model_path = 'modified_' + onnx_model_path
# onnx.save(model, new_model_path)

# print(f"Modified model saved to {new_model_path}")

# 加载模型并创建推理会话
# sess = ort.InferenceSession('modified_thor_segm0103.onnx')
# input_name = sess.get_inputs()[0].name

# 准备输入数据
# data_sample = datas[0]  

# # 运行模型并获取 ReduceMax 节点之前的输出
# outputs = sess.run(reduce_max_output_names, {input_name: data_sample})
# print(f"ReduceMax output: {outputs}")

# # 检查 ReduceMax 输出的形状和值
# print(f"Output shape: {outputs[0].shape}")
# print(f"Output max value: {np.max(outputs[0])}")
# print(f"Output min value: {np.min(outputs[0])}")

# # 获取输入名称
# input_names = [input.name for input in sess.get_inputs()]
# print("Input names:", input_names)

# # 获取输出名称
# output_names = [output.name for output in sess.get_outputs()]
# print("Output names:", output_names)


data_reader = DataReader(datas, 1)
# Quantize the model
# ... existing code ...

# 首先分析模型结构
def get_node_info(model):
    node_info = {}
    for node in model.graph.node:
        node_info[node.name] = {
            'op_type': node.op_type,
            'input': list(node.input),
            'output': list(node.output)
        }
    return node_info

# 获取所有需要排除的节点
def get_nodes_to_exclude(model):
    exclude_nodes = set()
    exclude_ops = {'ReduceMax', 'Resize', 'Constant', 'Shape', 'Gather', 'Unsqueeze'}
    
    for node in model.graph.node:
        if node.op_type in exclude_ops:
            exclude_nodes.add(node.name)
            # 添加相关联的节点
            for input_name in node.input:
                exclude_nodes.add(input_name)
            for output_name in node.output:
                exclude_nodes.add(output_name)
            
    return list(exclude_nodes)

# 分析模型并获取需要排除的节点
nodes_to_exclude = get_nodes_to_exclude(onnx_model)


# 打印所有节点类型
print("模型中的所有操作类型：")
op_types = set()
for node in onnx_model.graph.node:
    op_types.add(node.op_type)
print(op_types)

# 详细打印每个节点的信息
print("\n所有节点的详细信息：")
for node in onnx_model.graph.node:
    print(f"节点名称: {node.name}")
    print(f"操作类型: {node.op_type}")
    print(f"输入: {node.input}")
    print(f"输出: {node.output}")
    print("-" * 50)


# 使用更保守的量化设置
# 1. 首先定义所有需要特殊处理的操作
special_ops = {
    'ReduceMax', 
    'ReduceMean', 
    'ArgMax', 
    'Resize',
    'Div',
    'Sub',
    'Mul'
}

# 2. 收集所有相关节点的完整路径
special_nodes = set()
related_nodes = set()

for node in onnx_model.graph.node:
    if node.op_type in special_ops:
        # 收集节点本身
        special_nodes.add(node.name)
        # 收集输入输出
        for input_name in node.input:
            special_nodes.add(input_name)
        for output_name in node.output:
            special_nodes.add(output_name)
        
        # 收集相关联的节点
        for other_node in onnx_model.graph.node:
            if any(out in node.input for out in other_node.output):
                related_nodes.add(other_node.name)

# # 3. 使用更新后的量化配置
# quantize_static(
#     model_input=onnx_model_path,
#     model_output=model_quant_static_path,
#     calibration_data_reader=data_reader,
#     quant_format=QuantFormat.QDQ,
#     activation_type=QuantType.QUInt8,
#     weight_type=QuantType.QUInt8,
#     calibrate_method=CalibrationMethod.MinMax,
#     extra_options={
#         'NodesToExclude': list(special_nodes | related_nodes),
#         'OpTypesToExclude': list(special_ops),
#         'ForceQuantizeNoInputCheck': False,
#         'ActivationSymmetric': False,
#         'WeightSymmetric': False,
#         'EnableSubgraph': True,
#         'DedicatedQDQPair': True,  # 为每个tensor使用专用的QDQ对
#         'QDQOpTypePerChannel': ['Conv'],  # 为卷积层使用per-channel量化
#     }
# )

# 简化版本的量化代码
# quantize_static(
#     model_input=onnx_model_path, 
#     model_output=model_quant_static_path, 
#     calibration_data_reader=data_reader,
#     quant_format=QuantFormat.QDQ,
#     activation_type=QuantType.QUInt8,
#     weight_type=QuantType.QUInt8,
#     calibrate_method=CalibrationMethod.MinMax
# )

# 使用动态量化 可以实现
# from onnxruntime.quantization import quantize_dynamic

# quantize_dynamic(
#     model_input=onnx_model_path,
#     model_output=model_quant_static_path,
#     weight_type=QuantType.QUInt8
# )
#####################

import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationMethod

# 1. 首先分析模型中的所有 ReduceMax 相关节点
def find_nodes_to_exclude(model):
    nodes_to_exclude = set()
    
    for node in model.graph.node:
        # 找到所有 ReduceMax 节点
        if node.op_type == 'ReduceMax':
            print(f"Found ReduceMax node: {node.name}")
            # 添加节点本身
            nodes_to_exclude.add(node.name)
            # 添加其输入
            nodes_to_exclude.update(node.input)
            # 添加其输出
            nodes_to_exclude.update(node.output)
            
            # 找到与该节点相关的所有上下游节点
            for other_node in model.graph.node:
                # 如果当前节点的输出是其他节点的输入
                if any(output in other_node.input for output in node.output):
                    nodes_to_exclude.add(other_node.name)
                # 如果当前节点的输入是其他节点的输出
                if any(input in other_node.output for input in node.input):
                    nodes_to_exclude.add(other_node.name)

    return list(nodes_to_exclude)

# 2. 加载模型并获取需要排除的节点
model = onnx.load(onnx_model_path)
nodes_to_exclude = find_nodes_to_exclude(model)
print("Nodes to exclude:", nodes_to_exclude)

# 3. 使用更严格的量化配置
quantize_static(
    # 1. 基础参数
    model_input=onnx_model_path,
    model_output=model_quant_static_path,
    calibration_data_reader=data_reader,
    quant_format=QuantFormat.QDQ,        # 使用 QDQ 格式，在tensor上插入量化/反量化节点
    activation_type=QuantType.QUInt8,    # 激活值使用 UInt8，之前用的是 QInt8
    weight_type=QuantType.QUInt8,        # 权重使用 UInt8，之前用的是 QInt8
    calibrate_method=CalibrationMethod.MinMax,

    # 2. 重要改进：明确指定要量化的算子类型
    op_types_to_quantize=[
        'Conv',          # 卷积层
        'MatMul',        # 矩阵乘法
        'Gemm',          # 通用矩阵乘法
        'ConvTranspose'  # 反卷积层
    ],  # 之前没有明确指定，让量化器自己决定

    # 3. 明确指定要排除的节点
    nodes_to_exclude=nodes_to_exclude,   # 通过前面的分析得到的具体节点名称列表

    extra_options={
        # 4. 排除特定类型的算子
        # 'OpTypesToExclude': [
        #     'ReduceMax',    # 导致问题的算子
        #     'Resize',       # 可能影响精度的算子
        #     'Sigmoid',      # 激活函数
        #     'Constant',     # 常量节点
        #     'Reshape',      # 形状变换
        #     'Transpose'     # 转置操作
        # ],

        # 5. 量化策略设置
        'ActivationSymmetric': False,    # 激活值使用非对称量化
        'WeightSymmetric': True,         # 权重使用对称量化
        'DedicatedQDQPair': True,        # 为每个节点使用独立的QDQ对，提高精度
        'ForceQuantizeNoInputCheck': False,  # 不强制量化所有节点
        'EnableSubgraph': False,         # 禁用子图量化，避免复杂结构

        # 6. 新增的精度控制参数
        'QDQKeepRemovableActivations': True,  # 保留可移除的激活层
        'CalibMovingAverage': True,          # 使用移动平均提高稳定性
        'CalibMovingAverageConstant': 0.01,  # 移动平均系数
        'MinimumRealRange': 0.01,            # 设置最小量化范围

        # 7. 输出量化控制
        'OpTypesToExcludeOutputQuantization': [  # 不量化这些节点的输出
            'ReduceMax',
            'Resize',
            'Sigmoid'
        ]
    }
)
print('Quantization completed!')
