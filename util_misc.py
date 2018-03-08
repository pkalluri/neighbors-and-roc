

def force_tf_to_take_memory_only_as_needed():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    gpu_options = tf.GPUOptions(allow_growth=True)
    set_session(tf.Session(config=tf.ConfigProto(device_count={"GPU": 1}, gpu_options=gpu_options)))
