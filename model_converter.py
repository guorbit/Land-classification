import os
os.environ['TF_CPP_VMODULE'] = 'segment=2,convert_graph=2,convert_nodes=2,trt_engine=1,trt_logger=2'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tf2onnx
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from keras.models import load_model
from models.loss_constructor import SemanticLoss
from models.constructor import ModelGenerator,PyramidPoolingModule, UnetNanoConvBlock
from constants import (
    MODEL_NAME,
    MODELS,
    NUM_CLASSES,
    TEST_DATA_PATH,
    MODEL_ITERATION,
    LABEL_MAP,
    MODEL_FOLDER,
    TRAINING_DATA_PATH,
)
def tftrt_converter(onnx_model):
    # Convert the ONNX model to a TensorFlow model
    converter = trt.TrtGraphConverterV2(input_saved_model_dir='saved_model',
                                        conversion_params=trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                                            max_workspace_size_bytes=(1 << 32),
                                            precision_mode='FP32',
                                            maximum_cached_engines=100))
    converter.convert()
    converter.build(input_fn=build_input_fn)
    converter.save('saved_model_trt')
    return converter

def onnx_converter(frozen_graph):

    graph_def = frozen_graph.graph.as_graph_def()
    new_graph = tf.Graph()
    with new_graph.as_default():
        tf.import_graph_def(graph_def, name="")

    for op in new_graph.get_operations():
        print(op.name)
    # Convert the .pb file to ONNX
    model_proto, _ = tf2onnx.convert.from_graph_def(
        graph_def=graph_def,
        input_names=['x:0'],
        output_names=['vgg_nano_unet/activation_5/Softmax:0'],
        output_path='your_model.onnx',
        opset=11
    )

    return model_proto


def pb_converter(model):
    # Convert the Keras model to a ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Convert the ConcreteFunction to a TensorFlow GraphDef
    frozen_graph = convert_variables_to_constants_v2(full_model)

    # Save the model as a .pb file
    tf.io.write_graph(graph_or_graph_def=frozen_graph.graph,
                      logdir='.',
                      name='saved_model.pb',
                      as_text=False)  
    return frozen_graph


if __name__ == '__main__':
    loss = SemanticLoss(n_classes=NUM_CLASSES,weights_enabled=False)
    model = load_model(
                os.path.join(MODEL_FOLDER, MODEL_NAME + "_" + str(MODEL_ITERATION) + ".h5"),
                custom_objects={
                    "categorical_focal_loss": loss.categorical_focal_loss,
                    "categorical_jackard_loss": loss.categorical_jackard_loss,
                    "hybrid_loss": loss.hybrid_loss,
                    "categorical_ssim_loss": loss.categorical_ssim_loss,
                    "ModelGenerator": ModelGenerator,
                    "PyramidPoolingModule":PyramidPoolingModule,
                    "UnetNanoConvBlock":UnetNanoConvBlock,
                },
    )
    frozen_graph = pb_converter(model)
    onnx_model = onnx_converter(frozen_graph)


 


