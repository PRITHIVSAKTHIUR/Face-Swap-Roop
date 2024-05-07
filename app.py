# -* coding:UTF-8 -*
# !/usr/bin/env python
import numpy as np
import gradio as gr
import roop.globals
from roop.core import (
    start,
    decode_execution_providers,
    suggest_max_memory,
    suggest_execution_threads,
)
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import normalize_output_path
import os
from PIL import Image


def swap_face(source_file, target_file,doFaceEnhancer):

    source_path = "input.jpg"
    target_path = "target.jpg"

    source_image = Image.fromarray(source_file)
    source_image.save(source_path)
    target_image = Image.fromarray(target_file)
    target_image.save(target_path)

    print("source_path: ", source_path)
    print("target_path: ", target_path)

    roop.globals.source_path = source_path
    roop.globals.target_path = target_path
    output_path = "output.jpg"
    roop.globals.output_path = normalize_output_path(
        roop.globals.source_path, roop.globals.target_path, output_path
    )
    if doFaceEnhancer == True:
        roop.globals.frame_processors = ["face_swapper","face_enhancer"]
    else:
        roop.globals.frame_processors = ["face_swapper"]
    roop.globals.headless = True
    roop.globals.keep_fps = True
    roop.globals.keep_audio = True
    roop.globals.keep_frames = False
    roop.globals.many_faces = False
    roop.globals.video_encoder = "libx264"
    roop.globals.video_quality = 18
    roop.globals.max_memory = suggest_max_memory()
    roop.globals.execution_providers = decode_execution_providers(["cuda"])
    roop.globals.execution_threads = suggest_execution_threads()

    print(
        "start process",
        roop.globals.source_path,
        roop.globals.target_path,
        roop.globals.output_path,
    )

    for frame_processor in get_frame_processors_modules(
        roop.globals.frame_processors
    ):
        if not frame_processor.pre_check():
            return

    start()
    return output_path


app = gr.Interface(
    fn=swap_face, inputs=[gr.Image(), gr.Image(),gr.Checkbox(label="face_enhancer?", info="do face enhancer?")], outputs="image"
)
app.launch()
