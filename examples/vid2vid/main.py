import os
import sys
from typing import Dict, Literal, Optional

import fire
import torch
from torchvision.datasets.video_utils import VideoClips
from torchvision.io import write_video
from tqdm import tqdm


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.wrapper import StreamDiffusionWrapper


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(
    input: str,
    output: str = os.path.join(CURRENT_DIR, "..", "..", "images", "outputs", "output.mp4"),
    model_id: str = "KBlueLeaf/kohaku-v2.1",
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "1girl with brown dog ears, thick frame glasses",
    scale: float = 1.0,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    use_denoising_batch: bool = True,
    enable_similar_image_filter: bool = True,
    seed: int = 2,
    frames_per_clip: int = 240,
):
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    input : str, optional
        The input video name to load images from.
    output : str, optional
        The output video name to save images to.
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {"LoRA_1" : 0.5 , "LoRA_2" : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    scale : float, optional
        The scale of the image, by default 1.0.
    acceleration : Literal["none", "xformers", "tensorrt"]
        The type of acceleration to use for image generation.
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default True.
    enable_similar_image_filter : bool, optional
        Whether to enable similar image filter or not,
        by default True.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    """

    video_clips = VideoClips([input], clip_length_in_frames=frames_per_clip, frames_between_clips=frames_per_clip)
    first_video_clip = video_clips.get_clip(0)[0] / 255

    fps = video_clips.video_fps[0]
    # THWC
    height = int(first_video_clip.shape[1] * scale)
    width = int(first_video_clip.shape[2] * scale)

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id,
        lora_dict=lora_dict,
        t_index_list=[35, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        do_add_noise=False,
        mode="img2img",
        output_type="pt",
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=0.98,
        use_denoising_batch=use_denoising_batch,
        seed=seed,
    )

    stream.prepare(
        prompt=prompt,
        num_inference_steps=50,
    )

    # output frames size may slightly differ from input frames size
    # like 854x480 -> 848x480 (WxH)
    last_streamed_image: torch.Tensor = None
    for _ in range(stream.batch_size):
        last_streamed_image = stream(image=first_video_clip[0].permute(2, 0, 1))
    # recalculate the actual height and width
    height = last_streamed_image.shape[1]
    width = last_streamed_image.shape[2]
    for n in range(video_clips.num_clips()):
        video = video_clips.get_clip(n)[0] / 255
        frames = video.shape[0]
        video_result = torch.zeros(frames, height, width, 3)

        for i in tqdm(range(frames)):
            output_image = stream(video[i].permute(2, 0, 1))
            video_result[i] = output_image.permute(1, 2, 0)

        video_result = video_result * 255
        # TODO: merge clips into one video
        write_video(f"clip.{n}.{output}", video_result[2:], fps=fps)


if __name__ == "__main__":
    fire.Fire(main)
