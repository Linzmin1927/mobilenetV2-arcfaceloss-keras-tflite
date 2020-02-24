python3 -m tensorflow.python.tools.optimize_for_inference \
        --input=./tmp_2020-02-23_172730/frozen_model.pb \
        --output=./mobilenet_arcface_optimized.pb \
        --input_names="input_1" \
        --output_names="embeddings/l2_normalize"




