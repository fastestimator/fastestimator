import tensorflow as tf
import subprocess

def get_num_GPU():
    """
    Gets number of GPUs on device

    Returns:
        Number of GPUS available on device
    """
    try:
        result = subprocess.run(['nvidia-smi', '-q'], stdout=subprocess.PIPE).stdout.decode('utf-8')
        lines = [line.split() for line in result.splitlines() if line.startswith("Attached GPUs")]
        num_gpu = int(lines[0][-1])
    except:
        num_gpu = 0
    return num_gpu


def convert_tf_dtype(datatype):
    """
    Gets the tensorflow datatype from string
    
    Args:
        datatype: String of datatype

    Returns:
        Tensor data type
    """
    datatype_map = {"string": tf.string,
                    "int8": tf.int8,
                    "uint8": tf.uint8,
                    "int16": tf.int16,
                    "uint16": tf.uint16,
                    "int32": tf.int32,
                    "uint32": tf.uint32,
                    "int64": tf.int64,
                    "uint64": tf.uint64,
                    "float16": tf.float16,
                    "float32": tf.float32,
                    "float64": tf.float64}
    return datatype_map[datatype]
