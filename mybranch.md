```test_copy.py
from PIL import Image
from decord import VideoReader, cpu
import os

model = None
tokenizer = None

MAX_NUM_FRAMES=64 # if cuda OOM set a smaller number

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = 6400
    # print('vr.get_avg_fps()', vr.get_avg_fps())
    sample_fps = 10  # FPS
    print('sample_fps',sample_fps)
    # print('vr',len(vr))
    frame_idx = [i for i in range(0, vr, sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

def main(video_path):
    global model
    global tokenizer
    frames = encode_video(video_path)


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    img_path = os.path.join(dirname, 'background.mp4')
    main(img_path)
```


```test.py
import torch
from PIL import Image
from modelscope import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
import os

model = None
tokenizer = None

MAX_NUM_FRAMES=64 # if cuda OOM set a smaller number


def init():
    global model
    global tokenizer
    if model is None:
        model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)
        model = model.eval().cuda()
        return 'inited'
    else:
        return 'already inited'

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

def main(video_path):
    global model
    global tokenizer
    frames = encode_video(video_path)
    question = "描述这个视频"
    msgs = [
        {'role': 'user', 'content': frames + [question]},
    ]

    # Set decode params for video
    params = {}
    params["use_image_id"] = False
    params["max_slice_nums"] = 1 # use 1 if cuda OOM and video resolution > 448*448

    answer = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
        **params
    )
    print(answer)
    return answer

if __name__ == '__main__':
    init()
    init()
    dirname = os.path.dirname(__file__)
    img_path = os.path.join(dirname, 'background.mp4')
    img_path = os.path.join(dirname, 'fade-up-lines.gif.mp4')
    main(img_path)

```


```
huggingface-cli scan-cache -vvv
pip install huggingface_hub[cli]
huggingface-cli delete-cache // enter选中

```
