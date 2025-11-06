# -*- coding: utf-8 -*-
"""
ROS2 TTS 노드 (edge-tts + playsound)
- /tool_chat/out 구독 → 합성 후 재생
"""
import os, uuid, shutil, asyncio, threading, subprocess
from typing import Optional
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import edge_tts, playsound

class EdgeTTSWorker:
    def __init__(self, voice: str, rate: str, logger):
        self.voice, self.rate, self.logger = voice, rate, logger
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._q: Optional[asyncio.Queue] = None
        self._start_loop_thread()

    def _start_loop_thread(self):
        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._q = asyncio.Queue()
            try:
                self._loop.run_until_complete(self._consumer())
            except Exception as e:
                self.logger.error(f"[EdgeTTSWorker] loop error: {e}")
            finally:
                self._loop.stop()
                self._loop.close()
        self._thread = threading.Thread(target=_run, daemon=True); self._thread.start()

    async def _consumer(self):
        self.logger.info("[EdgeTTSWorker] consumer 시작")
        while not self._stop_event.is_set():
            try:
                text = await self._q.get()
                await self._speak_once(text)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"[EdgeTTSWorker] 소비 중 오류: {e}")
        self.logger.info("[EdgeTTSWorker] consumer 종료")

    def say(self, text: str):
        if not text: return
        if self._loop and self._q:
            asyncio.run_coroutine_threadsafe(self._q.put(text), self._loop).result()

    def stop(self):
        self._stop_event.set()
        try:
            if self._loop and self._q:
                asyncio.run_coroutine_threadsafe(self._q.put(""), self._loop).result()
        except Exception:
            pass
        if self._thread: self._thread.join(timeout=2.0)

    async def _speak_once(self, text: str):
        text = text.strip()
        if not text: return
        filename = f"{uuid.uuid4()}.mp3"
        try:
            self.logger.info(f"[TTS] 합성 시작: '{text}'")
            com = edge_tts.Communicate(text=text, voice=self.voice, rate=self.rate)
            await com.save(filename)
            try:
                await asyncio.to_thread(playsound.playsound, filename)
            except Exception as e:
                self.logger.warning(f"[TTS] playsound 실패, 폴백: {e}")
                self._fallback_play(filename)
            self.logger.info("[TTS] 재생 완료")
        except Exception as e:
            self.logger.error(f"[TTS] 합성/재생 오류: {e}")
        finally:
            try:
                if os.path.exists(filename): os.remove(filename)
            except Exception:
                pass

    def _fallback_play(self, filename: str):
        mpg123 = shutil.which("mpg123"); ffplay = shutil.which("ffplay")
        if mpg123:
            try: subprocess.run([mpg123, "-q", filename], check=True); return
            except Exception as e: self.logger.warning(f"[TTS] mpg123 실패: {e}")
        if ffplay:
            try: subprocess.run([ffplay, "-autoexit", "-nodisp", "-loglevel", "warning", filename], check=True); return
            except Exception as e: self.logger.warning(f"[TTS] ffplay 실패: {e}")
        self.logger.error("[TTS] 사용 가능한 재생기가 없습니다.")

class TTSEdgeNode(Node):
    def __init__(self):
        super().__init__("tts_edge_node")
        self.declare_parameter("voice", "ko-KR-InJoonNeural")
        self.declare_parameter("rate", "+5%")
        self.declare_parameter("topic", "/tool_chat/out")
        voice = self.get_parameter("voice").get_parameter_value().string_value
        rate  = self.get_parameter("rate").get_parameter_value().string_value
        topic = self.get_parameter("topic").get_parameter_value().string_value
        self.get_logger().info(f"[Param] voice={voice}, rate={rate}, topic={topic}")
        self._worker = EdgeTTSWorker(voice=voice, rate=rate, logger=self.get_logger())
        self._sub = self.create_subscription(String, topic, self._on_text, 10)
        self.get_logger().info(f"구독 시작: {topic}")

    def _on_text(self, msg: String):
        text = msg.data or ""
        self.get_logger().info(f"수신: {text}")
        self._worker.say(text)

    def destroy_node(self):
        try: self._worker.stop()
        except Exception: pass
        super().destroy_node()

def main():
    rclpy.init()
    node = TTSEdgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == "__main__":
    main()
