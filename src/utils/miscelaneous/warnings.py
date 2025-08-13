def turn_off_warnings():
    import absl.logging, logging, tensorflow as tf
    # Abseil → only ERROR+
    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold("error")

    # TensorFlow Python‑side logger → only ERROR+
    tf.get_logger().setLevel(logging.ERROR)
    # disable autograph/graph‑building logs
    tf.autograph.set_verbosity(0)

   