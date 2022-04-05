import os
os.system('python setup.py build develop')
os.system('gdown -O output/mixtrain/ 1XQsikiNY7ILgZvmvOeUf9oPDG4fTp0zs')

import cv2
import pandas as pd
import gradio as gr
from tools.demo import TextDemo
from maskrcnn_benchmark.config import cfg


def infer(filepath):
    cfg.merge_from_file('configs/mixtrain/seg_rec_poly_fuse_feature.yaml')
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

    text_demo = TextDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        output_polygon=True
    )
    image = cv2.imread(filepath)
    result_polygons, result_words = text_demo.run_on_opencv_image(image)
    text_demo.visualization(image, result_polygons, result_words)
    cv2.imwrite('result.jpg', image)
    return 'result.jpg', pd.DataFrame(result_words)


iface = gr.Interface(
    fn=infer,
    title="Mask TextSpotter v3",
    description="Mask TextSpotter v3 is an end-to-end trainable scene text spotter that adopts a Segmentation Proposal Network (SPN) instead of an RPN. Mask TextSpotter v3 significantly improves robustness to rotations, aspect ratios, and shapes.",
    inputs=[gr.inputs.Image(label="image", type="filepath")],
    outputs=[gr.outputs.Image(), gr.outputs.Dataframe(headers=['word'])],
    examples=['example1.jpg', 'example2.jpg', 'example3.jpg'],
    article="<a href=\"https://github.com/MhLiao/MaskTextSpotterV3\">GitHub Repo</a>",
).launch(enable_queue=True, cache_examples=True)
