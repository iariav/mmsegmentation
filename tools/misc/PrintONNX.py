import onnx


model = onnx.load('/home/iariav/Deep/Pytorch/mmsegmentation/work_dirs/b1_sgd_pixelformer_thermal_win7/iter_360000_opset15.onnx')
onnx.checker.check_model(model)
onnx.helper.printable_graph(model.graph)
####print a human readable representation of the graph
print('Model :\n\n{}'.format(onnx.helper.printable_graph(model.graph)))