from __future__ import annotations

import os
from threading import Lock

import kornia
import numpy as np
import torch
import torch.nn.functional as F
import vapoursynth as vs

from .ddcolor_arch import DDColor

__version__ = "1.0.0"

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")


@torch.inference_mode()
def ddcolor(
    clip: vs.VideoNode, device_index: int | None = None, num_streams: int = 1, model: int = 1, input_size: int = 512
) -> vs.VideoNode:
    """Towards Photo-Realistic Image Colorization via Dual Decoders

    :param clip:                    Clip to process. Only RGBH and RGBS formats are supported. RGBH uses the bfloat16
                                    data type for inference while RGBS uses the float32 data type.
    :param device_index:            Device ordinal of the GPU.
    :param num_streams:             Number of CUDA streams to enqueue the kernels.
    :param model:                   Model to use.
                                    0 = ddcolor_modelscope
                                    1 = ddcolor_artistic
    :param input_size:              Input size for model.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("ddcolor: this is not a clip")

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("ddcolor: only RGBH and RGBS formats are supported")

    if not torch.cuda.is_available():
        raise vs.Error("ddcolor: CUDA is not available")

    if num_streams < 1:
        raise vs.Error("ddcolor: num_streams must be at least 1")

    if model not in range(2):
        raise vs.Error("ddcolor: model must be 0 or 1")

    if os.path.getsize(os.path.join(model_dir, "ddcolor_artistic.pth")) == 0:
        raise vs.Error("ddcolor: model files have not been downloaded. run 'python -m vsddcolor' first")

    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda", device_index)

    stream = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    stream_lock = [Lock() for _ in range(num_streams)]

    match model:
        case 0:
            model_name = "ddcolor_modelscope.pth"
        case 1:
            model_name = "ddcolor_artistic.pth"

    state_dict = torch.load(os.path.join(model_dir, model_name), map_location="cpu")["params"]

    module = DDColor(input_size=(input_size, input_size), num_output_channels=2, last_norm="Spectral", num_queries=100)
    module.load_state_dict(state_dict, strict=False)
    module.eval().to(device, memory_format=torch.channels_last)
    if clip.format.bits_per_sample == 16:
        module.bfloat16()

    index = -1
    index_lock = Lock()

    @torch.inference_mode()
    def inference(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        nonlocal index
        with index_lock:
            index = (index + 1) % num_streams
            local_index = index

        with stream_lock[local_index], torch.cuda.stream(stream[local_index]):
            img = frame_to_tensor(f, device)
            orig_l = kornia.color.rgb_to_lab(img)[:, :1, :, :]

            img = F.interpolate(img, (input_size, input_size), mode="bilinear")
            img_l = kornia.color.rgb_to_lab(img)[:, :1, :, :]
            img_gray_lab = torch.cat([img_l, torch.zeros_like(img_l), torch.zeros_like(img_l)], dim=1)
            img_gray_rgb = kornia.color.lab_to_rgb(img_gray_lab)

            output_ab = module(img_gray_rgb)

            output_ab_resize = F.interpolate(output_ab, (clip.height, clip.width), mode="bilinear")
            output_lab = torch.cat([orig_l, output_ab_resize], dim=1)
            output = kornia.color.lab_to_rgb(output_lab)

            return tensor_to_frame(output, f.copy())

    return clip.std.FrameEval(lambda n: clip.std.ModifyFrame(clip, inference), clip_src=clip)


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    array = np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    tensor = torch.from_numpy(array).unsqueeze(0).to(device, memory_format=torch.channels_last)
    if tensor.dtype == torch.half:
        tensor = tensor.bfloat16()
    return tensor.clamp(0.0, 1.0)


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame) -> vs.VideoFrame:
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.half()
    array = tensor.squeeze(0).detach().cpu().numpy()
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane, :, :])
    return frame
