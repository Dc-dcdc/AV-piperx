"""Quest 相关的发送端工具。

当前负责把 MuJoCo 渲染出的图像编码成 JPEG，再通过 UDP 发回 Unity。
"""

from __future__ import annotations

import socket
import struct
import queue
import threading
import time

import cv2
import numpy as np

__all__ = ["UnityImageStreamer"]


class UnityImageStreamer:
    """Send MuJoCo camera frames to Unity as chunked JPEG UDP packets."""

    MAGIC = b"ZIMG"
    HEADER = struct.Struct("<4sIHHI")
    BROADCAST_HOST = "255.255.255.255"

    def __init__(
        self,
        host: str,
        port: int,
        send_hz: float,
        jpeg_quality: int,
        chunk_size: int,
        log_interval: float,
    ) -> None:
        host = str(host).strip()
        host_lower = host.lower()
        self.port = int(port)
        self.auto_host = host_lower == "auto"
        self.broadcast_host = host_lower in {"broadcast", self.BROADCAST_HOST}
        self.send_interval = 1.0 / send_hz if send_hz > 0.0 else 0.0
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self.chunk_size = int(np.clip(chunk_size, 256, 60000))
        self.log_interval = max(0.0, float(log_interval))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.frame_id = 0
        self.last_enqueue_t = 0.0
        self.last_send_t = 0.0
        self.last_error_t = 0.0
        self.last_log_t = 0.0
        self.sent_frames = 0
        self.sent_packets = 0
        self.sent_bytes = 0
        self.ack_count = 0
        self.last_ack_text = ""
        self.last_ack_addr = None
        self.last_ack_t = 0.0
        self._lock = threading.Lock()
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._closed = False

        if self.auto_host or self.broadcast_host:
            self.endpoint = (self.BROADCAST_HOST, self.port)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            if self.auto_host:
                print(f"Unity image target: broadcast:{self.port} until first Quest pose packet arrives")
            else:
                print(f"Unity image target: broadcast:{self.port}")
        else:
            self.endpoint = (host, self.port)
            print(f"Unity image target: {host}:{self.port}")

        self._thread = threading.Thread(target=self._send_loop, name="unity-image-streamer", daemon=True)
        self._thread.start()

    def update_auto_host(self, host: str) -> None:
        if not self.auto_host or not host:
            return
        endpoint = (host, self.port)
        changed = False
        with self._lock:
            if self.endpoint != endpoint:
                self.endpoint = endpoint
                changed = True
        if changed:
            print(f"Unity image auto target: {host}:{self.port}")

    def maybe_send_bgr(self, frame_bgr: np.ndarray) -> None:
        now = time.time()
        if self._closed:
            return
        if self.send_interval > 0.0 and now - self.last_enqueue_t < self.send_interval:
            return
        self.last_enqueue_t = now

        try:
            self._queue.put_nowait(frame_bgr)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame_bgr)
            except queue.Full:
                pass

    def _send_loop(self) -> None:
        while True:
            frame_bgr = self._queue.get()
            if frame_bgr is None:
                break
            self._send_bgr_now(frame_bgr)

    def _send_bgr_now(self, frame_bgr: np.ndarray) -> None:
        now = time.time()
        ok, encoded = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return

        payload = encoded.tobytes()
        chunk_count = int(np.ceil(len(payload) / self.chunk_size))
        if chunk_count <= 0 or chunk_count > 65535:
            if now - self.last_error_t > 1.0:
                print(f"Unity image frame too large to send: {len(payload)} bytes, chunks={chunk_count}")
                self.last_error_t = now
            return

        frame_id = self.frame_id
        total_bytes = len(payload)
        packets_sent = 0
        try:
            with self._lock:
                endpoint = self.endpoint
            for chunk_index in range(chunk_count):
                start = chunk_index * self.chunk_size
                chunk = payload[start : start + self.chunk_size]
                header = self.HEADER.pack(self.MAGIC, frame_id, chunk_index, chunk_count, total_bytes)
                self.sock.sendto(header + chunk, endpoint)
                packets_sent += 1
            self.frame_id = (self.frame_id + 1) & 0xFFFFFFFF
            self.last_send_t = now
            self.sent_frames += 1
            self.sent_packets += packets_sent
            self.sent_bytes += total_bytes
            self._drain_acks(now)
            self._maybe_log(now, total_bytes, chunk_count)
        except OSError as exc:
            if now - self.last_error_t > 1.0:
                print(f"Failed to send Unity image frame to {endpoint[0]}:{endpoint[1]}: {exc}")
                self.last_error_t = now

    def _drain_acks(self, now: float) -> None:
        while True:
            try:
                data, addr = self.sock.recvfrom(4096)
            except (BlockingIOError, socket.timeout):
                break
            except OSError as exc:
                if now - self.last_error_t > 1.0:
                    print(f"Unity image ACK receive error: {exc}")
                    self.last_error_t = now
                break

            text = data.decode("utf-8", errors="replace").strip()
            if not text.startswith("ZACK"):
                continue
            self.ack_count += 1
            self.last_ack_text = text
            self.last_ack_addr = addr
            self.last_ack_t = now
            print(f"Unity image ack: from={addr[0]}:{addr[1]} {text}")

    def _maybe_log(self, now: float, frame_bytes: int, chunk_count: int) -> None:
        if self.log_interval <= 0.0 or now - self.last_log_t < self.log_interval:
            return
        self._drain_acks(now)
        target = f"{self.endpoint[0]}:{self.endpoint[1]}"
        if self.last_ack_addr is None:
            ack_text = "ack=none"
        else:
            ack_age = now - self.last_ack_t
            ack_text = f"ack={self.ack_count} last_from={self.last_ack_addr[0]}:{self.last_ack_addr[1]} age={ack_age:.1f}s"
        print(
            f"Unity image sent: target={target}, frame={self.sent_frames}, "
            f"jpeg={frame_bytes} bytes, chunks={chunk_count}, total_packets={self.sent_packets}, {ack_text}"
        )
        self.last_log_t = now

    def close(self) -> None:
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        self._thread.join(timeout=1.0)
        self.sock.close()
