import tensorflow as tf

# Load model
loaded = tf.saved_model.load('/tmp/mlp_model/mlp_burn_severity')
serve = loaded.signatures['serving_default']

FEATURE_NAMES = [
    'dnbr', 'dndvi', 'dndbi', 'dbsi', 'nbr', 'bsi', 'ndvi', 'ndbi',
    'meanelev_32', 'wc_bio19', 'nirBand', 'wc_bio05', 'rdgh_6', 'blueBand',
    'minelev_4', 'greenBand', 'wc_bio06', 'swir2Band', 'pisrdif_2021-11-22',
    'pisrdif_2021-12-22', 'stddevelev_32', 'maxc_2', 'wc_bio12', 'wc_bio07',
    'dmndwi', 'wc_bio18', 'wc_bio17', 'wc_bio02', 'vd_5', 'planc_32'
]

class GEEWrapper(tf.Module):
    def __init__(self, loaded_model):
        super().__init__()
        self.loaded = loaded_model
        self.serve = loaded_model.signatures['serving_default']

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32, name=name)
        for name in FEATURE_NAMES
    ])
    def __call__(self, *inputs):
        # Stack the individual band inputs into [batch, 30]
        # Each input is [height, width], flatten to [pixels, 1] then concat
        shape = tf.shape(inputs[0])
        h, w = shape[0], shape[1]
        flat_inputs = [tf.reshape(inp, [h * w, 1]) for inp in inputs]
        combined = tf.concat(flat_inputs, axis=1)  # [pixels, 30]
        result = self.serve(covariate_input=combined)
        # Reshape output back to [height, width, 4]
        output = tf.reshape(result['output_0'], [h, w, 4])
        return {'output': output}

wrapper = GEEWrapper(loaded)

tf.saved_model.save(
    wrapper,
    '/tmp/mlp_model_gee2/',
    signatures={'serving_default': wrapper.__call__}
)

print("Re-exported with named band inputs!")