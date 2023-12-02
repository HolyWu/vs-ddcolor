# DDColor
Towards Photo-Realistic Image Colorization via Dual Decoders, based on https://github.com/piddnad/DDColor.


## Dependencies
- [PyTorch](https://pytorch.org/get-started) 2.1.1 or later
- [VapourSynth](http://www.vapoursynth.com/) R62 or later


## Installation
```
pip install -U vsddcolor
python -m vsddcolor
```


## Usage
```python
from vsddcolor import ddcolor

ret = ddcolor(clip)
```

See `__init__.py` for the description of the parameters.
