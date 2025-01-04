import os
import onnx
import onnxruntime as ort
import numpy as np
import cv2
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_static, QuantType, CalibrationMethod
from onnxruntime import InferenceSession, get_available_providers

onnx_model_path = 'thor_segm0103.onnx'
model_quant_static_path = 'quant_model_99_0103.onnx'

img_dir = r'D:\Data\thor\thor_roi\20241213\images'
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
sess = ort.InferenceSession('modified_thor_segm0103.onnx')
input_name = sess.get_inputs()[0].name

# 准备输入数据
data_sample = datas[0]  

# 运行模型并获取 ReduceMax 节点之前的输出
outputs = sess.run(reduce_max_output_names, {input_name: data_sample})
print(f"ReduceMax output: {outputs}")

# 检查 ReduceMax 输出的形状和值
print(f"Output shape: {outputs[0].shape}")
print(f"Output max value: {np.max(outputs[0])}")
print(f"Output min value: {np.min(outputs[0])}")

# 获取输入名称
input_names = [input.name for input in sess.get_inputs()]
print("Input names:", input_names)

# 获取输出名称
output_names = [output.name for output in sess.get_outputs()]
print("Output names:", output_names)


data_reader = DataReader(datas, 1)
# Quantize the model
nodes_to_exclude = ['ReduceMax', 'Resize']
quantize_static(
    model_input=onnx_model_path, 
    model_output=model_quant_static_path, 
    # calibration_data_reader=data_reader,
    # quant_format=QuantFormat.QDQ, 
    activation_type=QuantType.QInt8, 
    weight_type=QuantType.QInt8,
    # nodes_to_exclude=nodes_to_exclude,
    calibrate_method=CalibrationMethod.MinMax,
    extra_options={'NodesToExclude': ['/head/conv_out/conv_out.2/Constant_1_output_0_ReduceMax', '/head/conv_out/conv_out.2', '/head/conv_out/conv_out.2/Constant_1','/bga/up2/Constant_1_output_0_ReduceMax',  # 具体出错的 ReduceMax 节点
    '/bga/up2/Resize',                         # Resize 节点
    '/bga/up2/Constant_1',                     # 相关常量节点
    '/bga/up2/Constant']}
   )


