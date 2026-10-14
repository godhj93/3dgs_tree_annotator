#!/usr/bin/env python3
"""Local annotation app and persistent, ROS-independent Depth Anything V2 worker."""
from __future__ import annotations

import argparse
import base64
import binascii
import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


class DepthRequest(BaseModel):
    frame_id: str = Field(min_length=1, max_length=160)
    rgb_png_base64: str = Field(min_length=1, max_length=12_000_000)


class DepthWorker:
    def __init__(self, source: Path, checkpoint: str | None, device: str):
        self.source = source
        self.checkpoint = checkpoint
        self.device_name = device
        self.model = None
        self.error = None
        self.lock = threading.Lock()

    def load(self):
        import torch
        from huggingface_hub import hf_hub_download

        if not (self.source / 'depth_anything_v2/dpt.py').is_file():
            raise RuntimeError(f'DA2 source not found: {self.source}')
        sys.path.insert(0, str(self.source))
        from depth_anything_v2.dpt import DepthAnythingV2
        from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize
        from torchvision.transforms import Compose

        device = self.device_name
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device_name = device
        path = self.checkpoint or hf_hub_download(
            repo_id='depth-anything/Depth-Anything-V2-Small',
            filename='depth_anything_v2_vits.pth',
        )
        model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        self.model = model.to(device).eval()
        # Same preprocessing as upstream infer_image, but honor the configured
        # device instead of upstream image2tensor's automatic CUDA device.
        self.transform = Compose([
            Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True,
                   ensure_multiple_of=14, resize_method='lower_bound',
                   image_interpolation_method=cv2.INTER_CUBIC),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    def infer(self, image: np.ndarray) -> np.ndarray:
        import torch
        from torch.nn import functional as F

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        tensor = torch.from_numpy(self.transform({'image': rgb})['image']).unsqueeze(0).to(self.device_name)
        with torch.inference_mode():
            predicted = self.model(tensor)
            predicted = F.interpolate(predicted[:, None], image.shape[:2], mode='bilinear', align_corners=True)[0, 0]
        return predicted.float().cpu().numpy()


def create_app(root: Path, worker: DepthWorker) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        try:
            worker.load()
        except Exception as error:
            worker.error = str(error)
            logging.exception('DA2 unavailable; static annotation app remains available')
        yield

    app = FastAPI(title='3DGS annotation + DA2', lifespan=lifespan)

    @app.get('/api/depth/status')
    def status():
        return {'ready': worker.model is not None and worker.error is None,
                'model': 'Depth-Anything-V2-Small', 'device': worker.device_name,
                'input_size': 518, 'error': worker.error}

    @app.post('/api/depth')
    def depth(request: DepthRequest):
        if worker.model is None or worker.error:
            raise HTTPException(503, detail=worker.error or 'Depth model is not ready')
        try:
            encoded = base64.b64decode(request.rgb_png_base64, validate=True)
        except (ValueError, binascii.Error):
            raise HTTPException(400, detail='Invalid PNG base64')
        if not encoded.startswith(b'\x89PNG\r\n\x1a\n'):
            raise HTTPException(400, detail='Expected a PNG image')
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (544, 963):
            raise HTTPException(400, detail='Expected a 963x544 sensor-view PNG')
        if not worker.lock.acquire(blocking=False):
            raise HTTPException(429, detail='Depth worker busy; retry the latest frame')
        try:
            started = time.perf_counter()
            raw = worker.infer(image)
            output = cv2.resize(raw, (640, 640), interpolation=cv2.INTER_LINEAR).astype('<f4')
            if not np.isfinite(output).all():
                raise RuntimeError('Depth model returned non-finite values')
            return {'frame_id': request.frame_id, 'width': 640, 'height': 640,
                    'dtype': '<f4', 'depth_f32_base64': base64.b64encode(output.tobytes()).decode('ascii'),
                    'model': 'Depth-Anything-V2-Small', 'input_size': 518,
                    'depth_kind': 'raw_da2_relative_output', 'units': 'relative',
                    'inference_ms': round((time.perf_counter() - started) * 1000, 2)}
        except Exception as error:
            logging.exception('Depth inference failed')
            raise HTTPException(503, detail=str(error))
        finally:
            worker.lock.release()

    app.mount('/', StaticFiles(directory=root, html=True), name='static')
    return app


def main():
    import uvicorn

    project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=project)
    parser.add_argument('--da2-source', type=Path, default=project / 'Depth-Anything-V2')
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8027)
    args = parser.parse_args()
    worker = DepthWorker(args.da2_source.resolve(), args.checkpoint, args.device)
    uvicorn.run(create_app(args.root.resolve(), worker), host=args.host, port=args.port, workers=1)


if __name__ == '__main__':
    main()
