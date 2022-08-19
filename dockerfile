FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN pip install psnr_hvsm scipy einops pytorch_msssim opencv-python-headless pybind11 \
    && pip install ptflops==0.6.5 \
    && pip install lmdb

WORKDIR /workspace

