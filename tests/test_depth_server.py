import base64
import importlib.util
from pathlib import Path
import threading
import unittest

import cv2
from fastapi.testclient import TestClient
import numpy as np

spec = importlib.util.spec_from_file_location('annotation_depth_server', Path(__file__).resolve().parents[1] / 'server.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class StubWorker:
    def __init__(self):
        self.model = object()
        self.error = None
        self.device_name = 'cpu'
        self.lock = threading.Lock()
        self.image = None

    def load(self):
        pass

    def infer(self, image):
        self.image = image
        return np.linspace(0, 2, 963, dtype=np.float32)[None].repeat(544, axis=0)


class DepthServerTests(unittest.TestCase):
    def setUp(self):
        self.worker = StubWorker()
        self.client = TestClient(server.create_app(Path(__file__).resolve().parents[1], self.worker))
        image = np.zeros((544, 963, 3), np.uint8)
        image[..., 0] = 35
        image[..., 2] = 210
        self.request = {'frame_id': 'frame_17', 'rgb_png_base64': base64.b64encode(cv2.imencode('.png', image)[1]).decode()}

    def test_frame_shape_precision_and_png_channel_order(self):
        result = self.client.post('/api/depth', json=self.request)
        self.assertEqual(result.status_code, 200)
        body = result.json()
        self.assertEqual(body['frame_id'], 'frame_17')
        self.assertEqual((body['height'], body['width'], body['dtype']), (640, 640, '<f4'))
        depth = np.frombuffer(base64.b64decode(body['depth_f32_base64']), dtype='<f4').reshape(640, 640)
        self.assertTrue(np.isfinite(depth).all())
        self.assertLess(depth[:, 0].mean(), .01)
        self.assertGreater(depth[:, -1].mean(), 1.99)
        self.assertEqual(self.worker.image[0, 0].tolist(), [35, 0, 210])

    def test_invalid_input_and_unavailable_worker(self):
        bad = dict(self.request, rgb_png_base64='!invalid!')
        self.assertEqual(self.client.post('/api/depth', json=bad).status_code, 400)
        self.worker.error = 'model unavailable'
        self.assertFalse(self.client.get('/api/depth/status').json()['ready'])
        self.assertEqual(self.client.post('/api/depth', json=self.request).status_code, 503)

    def test_wrong_sensor_dimensions(self):
        png = cv2.imencode('.png', np.zeros((640, 640, 3), np.uint8))[1]
        bad = dict(self.request, rgb_png_base64=base64.b64encode(png).decode())
        self.assertEqual(self.client.post('/api/depth', json=bad).status_code, 400)

    def test_busy_worker_does_not_queue_frames(self):
        self.worker.lock.acquire()
        try:
            self.assertEqual(self.client.post('/api/depth', json=self.request).status_code, 429)
        finally:
            self.worker.lock.release()
        self.assertEqual(self.client.post('/api/depth', json=self.request).status_code, 200)

    def test_inference_failure_releases_worker(self):
        self.worker.infer = lambda image: (_ for _ in ()).throw(RuntimeError('test inference failure'))
        with self.assertLogs(level='ERROR'):
            self.assertEqual(self.client.post('/api/depth', json=self.request).status_code, 503)
        self.assertTrue(self.worker.lock.acquire(blocking=False))
        self.worker.lock.release()


if __name__ == '__main__':
    unittest.main()
