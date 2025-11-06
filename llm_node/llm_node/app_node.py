# -*- coding: utf-8 -*-
"""
app_node.py
- Flask-SocketIO 웹 UI 서버
- ROS2 통합
- llm_inference (v5.11) 초간단 버전에 호환됨
- 'mode', 'tool_final', 'section' 3개 키만 사용
- 'LIGHT_ON' 모드 지원
- 'generate_speech' 함수 호출

추가 기능:
- 작업 상태 메모리 (active_mode, active_tools, last_task_desc 등)
- 이미 작업 중일 때 새 작업 모드 요청 → Busy 피드백
- DELIVER/RETURN 성공 시점은 /task/done 기준으로 도구 메모리 업데이트
- FINISH 시, 사용 중 도구가 있으면 "도구들을 모두 반납하고 종료할까요?" 협의
- "초기화해줘" → 상태 초기화
- "직전에 수행했던 기능이 뭐야" → last_task_desc 설명
"""

from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO, emit
import sys, threading, os, json, warnings, time, re
from typing import Optional, Dict, Any
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ➕ v5.11 임포트
from llm_inference import ToolChatEngine  # tool_chat_engine.py

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ----- Flask -----
BASE_DIR = Path("/home/liw/zeus_ws/src/llm_node/llm_node")
TEMPLATE_DIR = str(BASE_DIR / "templates")
STATIC_DIR   = str(BASE_DIR / "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR, static_url_path="/static")
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

print("TEMPLATE_DIR =", TEMPLATE_DIR)
print("STATIC_DIR   =", STATIC_DIR)

# ----- QoS -----
QOS_LATCHED = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
QOS_RELIABLE = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                          reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.VOLATILE)

# ----- 도구/모드 -----
NIPPER_SYNS = ["니퍼","니뻐","니빠","롱노즈","롱노즈 플라이어","롱노우즈 플라이어","롱 플라이어","롱플라이어",
               "needle nose","needle-nose","long nose","long-nose","nipper"]
TOOL_EN_MAP = {**{k: "nipper" for k in NIPPER_SYNS},
               "커터":"wire_cutter","와이어 스트리퍼":"wire_stripper","M3 나사":"M3","스탠드":"lamp",
               "wire_cutter":"wire_cutter","wire_stripper":"wire_stripper","M3":"M3","lamp":"lamp","NONE":"NONE"}
TOOL_KR_MAP = {"nipper":"니퍼","wire_cutter":"커터","wire_stripper":"와이어 스트리퍼","M3":"M3 나사","lamp":"스탠드","NONE":""}

# ALLOWED_MODES 6개로 유지 (RESET 등은 내부 처리로만 사용)
ALLOWED_MODES = {"START","DELIVER","RETURN","FINISH","INFO","LIGHT_ON","LIGHT_OFF"}

def _tool_kr_for_speech(tool_en: str) -> str:
    kr = TOOL_KR_MAP.get(tool_en, tool_en or "")
    return "엠쓰리 나사" if kr.upper().startswith("M3") else kr

def _has_batchim(k: str) -> bool:
    if not k: return False
    ch = k.strip()[-1]; code = ord(ch) - 0xAC00
    return 0 <= code <= 11171 and (code % 28) != 0

def _eulreul(word: str) -> str: return "을" if _has_batchim(word) else "를"


# ----- ROS2 노드 -----
class WebUINode(Node):
    def __init__(self, use_external_nodes=False):
        super().__init__('web_ui_node')
        self.use_external_nodes = use_external_nodes
        self._last_out_text = ""
        self._last_out_ts = 0.0

        # 작업/도구 상태 메모리
        self.active_mode: Optional[str] = None         # 현재 실행 중인 모드
        self.active_tool: str = "NONE"                 # 현재 실행 중인 도구(en)
        self.active_section: int = 0                   # 현재 실행 중인 섹션
        self.active_tools = set()                      # 현재 "밖에 있는" 도구(en) 집합
        self.last_tool_en: str = "NONE"                # 가장 최근 사용 도구(en)
        self.last_task_desc: str = ""                  # 가장 최근 수행 작업 설명
        self.pending_finish_with_return: bool = False  # FINISH 시 반납 여부 확인 중
        self.pending_auto_finish_after_return: bool = False  # RETURN 완료 후 자동 FINISH 예정

        if not use_external_nodes:
            self.declare_parameter('model_path', '/home/liw/sllm/models/gemma3-q4_k_m_budda.gguf')
            self.declare_parameter('mode_rag_jsonl_path', '/home/temp_id/ros2_ws/src/vision_node/rule.jsonl')
            self.declare_parameter('faiss_index_path', './faiss_index_ros')
            self.declare_parameter('use_gpu_llama', True)
            self.declare_parameter('engine_verbose', True)
            model_path = self.get_parameter('model_path').get_parameter_value().string_value
            mode_rag_jsonl = self.get_parameter('mode_rag_jsonl_path').get_parameter_value().string_value
            faiss_index_path = self.get_parameter('faiss_index_path').get_parameter_value().string_value
            use_gpu_llama = self.get_parameter('use_gpu_llama').get_parameter_value().bool_value
            engine_verbose = self.get_parameter('engine_verbose').get_parameter_value().bool_value
            self.get_logger().info('[LLM] ToolChatEngine 초기화 중...')
            self.chat_engine = ToolChatEngine(model_path=model_path, mode_rag_jsonl_path=mode_rag_jsonl,
                                              faiss_index_path=faiss_index_path, use_gpu_llama=use_gpu_llama,
                                              verbose=engine_verbose)
            self.get_logger().info('[LLM] ToolChatEngine 초기화 완료')
        else:
            self.chat_engine = None
            self.get_logger().info('[외부 노드] ros2_chat_node/tts_node 활용')

        self.control_pub = self.create_publisher(String, '/aiot/string/llm_cmd', QOS_RELIABLE)
        self.chat_in_pub = self.create_publisher(String, '/tool_chat/in', QOS_LATCHED)
        if not use_external_nodes:
            self.chat_out_pub = self.create_publisher(String, '/tool_chat/out', QOS_RELIABLE)

        self.chat_in_echo_sub = self.create_subscription(String, '/tool_chat/in',
                                                         self.chat_in_echo_callback, QOS_LATCHED)
        if not use_external_nodes:
            self.chat_in_sub = self.create_subscription(String, '/tool_chat/in',
                                                        self.chat_callback, QOS_LATCHED)

        self.chat_out_sub = self.create_subscription(String, '/tool_chat/out',
                                                     self.chat_out_callback, QOS_RELIABLE)
        self.tool_detection_sub = self.create_subscription(String, '/tool_detections',
                                                           self.tool_detection_callback, QOS_RELIABLE)
        self.tts_status_sub = self.create_subscription(String, '/tts/status',
                                                       self.tts_status_callback, QOS_RELIABLE)
        
        self.task_done_sub = self.create_subscription(
            String, '/task/done', self.task_done_callback, QOS_RELIABLE
        )

        self.last_tool_info = {}
        self.last_chat_response = ""
        self.tts_status = {"playing": False, "last_text": ""}

    def _reset_memory(self):
        """사용자 요청(초기화)나 FINISH 후에 상태 초기화."""
        self.active_mode = None
        self.active_tool = "NONE"
        self.active_section = 0
        self.active_tools = set()
        self.last_tool_en = "NONE"
        self.last_task_desc = ""
        self.pending_finish_with_return = False
        self.pending_auto_finish_after_return = False

    def _describe_current_task(self) -> str:
        """현재 active_mode/active_tool/section을 한국어로 요약."""
        mode = self.active_mode
        tool = self.active_tool or "NONE"
        sec = self.active_section or 0

        if not mode:
            return "작업"

        if mode == "DELIVER":
            if tool != "NONE":
                return f"{_tool_kr_for_speech(tool)} 전달 작업"
            return "도구 전달 작업"
        if mode == "RETURN":
            if tool == "NONE":
                return "도구 전체 반납 작업"
            return f"{_tool_kr_for_speech(tool)} 반납 작업"
        if mode == "LIGHT_ON":
            if sec:
                return f"{sec}번 섹션 조명 작업"
            return "조명 비추는 작업"
        if mode == "START":
            return "작업 시작 처리"
        if mode == "FINISH":
            return "작업 종료 처리"
        return "작업"

    def task_done_callback(self, msg: String):
        text = msg.data or ""
        now = time.time()

        # 현재 active_mode/active_tool 기준으로 작업 결과 업데이트
        mode = self.active_mode
        tool = self.active_tool or "NONE"
        sec = self.active_section or 0

        if mode == "DELIVER":
            if tool != "NONE":
                self.active_tools.add(tool)
                self.last_tool_en = tool
                self.last_task_desc = f"DELIVER: {_tool_kr_for_speech(tool)} 전달"
        elif mode == "RETURN":
            if tool == "NONE":
                # 전체 반납
                self.active_tools = set()
                self.last_task_desc = "RETURN: 모든 도구 반납"
            else:
                self.active_tools.discard(tool)
                self.last_tool_en = tool
                self.last_task_desc = f"RETURN: {_tool_kr_for_speech(tool)} 반납"
        elif mode == "LIGHT_ON":
            self.last_task_desc = f"LIGHT_ON: {sec}번 섹션 비춤"
        
        elif mode == "LIGHT_OFF":
            self.last_task_desc = f"LIGHT_OFF: {sec}번 섹션 끔"
        
        elif mode == "START":
            self.last_task_desc = "START: 작업 시작"
        
        elif mode == "FINISH":
            self.last_task_desc = "FINISH: 작업 종료"
            # FINISH 시 전체 상태 초기화
            self._reset_memory()

        # RETURN 완료 후 자동 FINISH 예약이 켜져 있으면 FINISH 명령 발행
        if self.pending_auto_finish_after_return and mode == "RETURN":
            self.pending_auto_finish_after_return = False
            # FINISH 명령 발행
            try:
                payload = {"mode": "FINISH", "tool": "NONE", "section": 0}
                out = String()
                out.data = json.dumps(payload, ensure_ascii=False)
                self.control_pub.publish(out)
                self.get_logger().info(f"[CONTROL-AUTO-FINISH] {payload}")
                # FINISH를 현재 active 작업으로 설정
                self.active_mode = "FINISH"
                self.active_tool = "NONE"
                self.active_section = 0
            except Exception as e:
                self.get_logger().error(f"자동 FINISH 발행 실패: {e}")

        # 현재 명령 종료 처리 (단, 위의 FINISH 자동 발행에서 다시 설정될 수 있음)
        if mode not in (None, "FINISH"):
            self.active_mode = None
            self.active_tool = "NONE"
            self.active_section = 0
            self.pending_finish_with_return = False

        socketio.emit('task_done', {'text': text, 'timestamp': now})

    # ---- 콜백 ----
    def chat_in_echo_callback(self, msg: String):
        text = msg.data or ""
        socketio.emit('incoming_text', {'text': text, 'timestamp': time.time()})

    def chat_callback(self, msg: String):
        if self.use_external_nodes: return
        user_input = (msg.data or "").strip()
        if not user_input: return
        self.get_logger().info(f"[IN] {user_input}")
        _ = self.process_chat_message(user_input)

    def chat_out_callback(self, msg: String):
        now = time.time()
        text = msg.data or ""
        if text == self._last_out_text and (now - self._last_out_ts) < 0.6:
            return
        self._last_out_text, self._last_out_ts = text, now
        self.last_chat_response = text
        self.get_logger().info(f'[응답 수신] {text}')
        socketio.emit('external_chat_response', {'response': text, 'timestamp': now})

    def tool_detection_callback(self, msg: String):
        try:
            tool_data = json.loads(msg.data)
            self.last_tool_info = tool_data
            socketio.emit('tool_detection', tool_data)
        except Exception as e:
            self.get_logger().warning(f'도구 감지 데이터 파싱 실패: {e}')

    def tts_status_callback(self, msg: String):
        try:
            status_data = json.loads(msg.data)
            self.tts_status.update(status_data)
            socketio.emit('tts_status', {'status': self.tts_status,'timestamp': time.time()})
        except Exception as e:
            self.get_logger().warning(f'TTS 상태 데이터 파싱 실패: {e}')

    # ---- 처리 ----
    def _to_en_tool(self, tool: str) -> str:
        if not tool: return "NONE"
        return TOOL_EN_MAP.get(tool.strip(), tool)

    def _speak(self, text: str):
        """내장 TTS 노드로 한 줄 보내기."""
        if hasattr(self, "chat_out_pub"):
            m = String()
            m.data = text
            self.chat_out_pub.publish(m)

    def _is_yes(self, text_norm: str) -> bool:
        return any(k in text_norm for k in ["네", "예", "응", "어", "그래", "맞아", "좋아", "yes", "y"])

    def _is_no(self, text_norm: str) -> bool:
        return any(k in text_norm for k in ["아니", "아니요", "싫어", "no", "노", "n"])

    def _is_reset_command(self, text_norm: str) -> bool:
        return any(k in text_norm for k in ["초기화", "리셋", "상태 초기화", "다 잊어", "다 잊어줘"])

    def _is_last_task_query(self, text_norm: str) -> bool:
        # "직전/방금/바로 전/마지막" + "작업/기능/뭐 했/무슨 일"
        if not any(k in text_norm for k in ["직전", "방금", "바로 전", "바로전", "마지막"]):
            return False
        return any(k in text_norm for k in ["작업", "기능", "뭐 했", "뭐했", "무슨 일", "무슨일"])

    # process_chat_message (v5.11 호환 + 협업 로직)
    def process_chat_message(self, user_input: str) -> Dict[str, Any]:
        # 0) 공백 제거 및 소문자 버전
        raw_text = (user_input or "").strip()
        text_norm = raw_text.lower()

        # 0-1) FINISH + RETURN 협의 중일 때 (네/아니오) 우선 처리
        if self.pending_finish_with_return:
            if self._is_yes(text_norm):
                # YES: 도구 모두 반납 후 자동 FINISH
                self.pending_finish_with_return = False
                self.pending_auto_finish_after_return = True

                desc = "현재 사용 중인 도구들을 모두 반납한 뒤 작업을 종료하겠습니다."
                try:
                    speech = self.chat_engine.generate_speech(
                        task_description=desc,
                        style="친절하고 간결하게"
                    )
                except Exception:
                    speech = "도구들을 모두 반납하고 작업을 종료하겠습니다."

                # RETURN(전체) 명령 발행 (tool="NONE" → 전체 반납)
                payload = {"mode": "RETURN", "tool": "NONE", "section": 0}
                msg = String()
                msg.data = json.dumps(payload, ensure_ascii=False)
                self.control_pub.publish(msg)
                self.get_logger().info(f"[CONTROL-RETURN-ALL] {payload}")

                # 현재 실행 중 작업으로 RETURN 설정
                self.active_mode = "RETURN"
                self.active_tool = "NONE"
                self.active_section = 0

                self._speak(speech)
                return {'mode': 'RETURN_ALL_THEN_FINISH', 'tool': 'NONE', 'section': 0, 'nlg': speech}

            if self._is_no(text_norm):
                # NO: 바로 FINISH (반납 없이 종료)
                self.pending_finish_with_return = False
                self.pending_auto_finish_after_return = False

                desc = "도구 반납 없이 작업을 종료하겠습니다."
                try:
                    speech = self.chat_engine.generate_speech(
                        task_description=desc,
                        style="친절하고 간결하게"
                    )
                except Exception:
                    speech = "도구 반납 없이 작업을 종료하겠습니다."

                payload = {"mode": "FINISH", "tool": "NONE", "section": 0}
                msg = String()
                msg.data = json.dumps(payload, ensure_ascii=False)
                self.control_pub.publish(msg)
                self.get_logger().info(f"[CONTROL-FINISH-NO-RETURN] {payload}")

                self.active_mode = "FINISH"
                self.active_tool = "NONE"
                self.active_section = 0

                self._speak(speech)
                return {'mode': 'FINISH_NO_RETURN', 'tool': 'NONE', 'section': 0, 'nlg': speech}

            # 네/아니오가 아닌 경우 → 다시 요청
            speech = "도구들을 모두 반납하고 종료할지 여부를 네 또는 아니오로 말씀해 주세요."
            self._speak(speech)
            return {'mode': 'ASK_FINISH_CONFIRM', 'tool': 'NONE', 'section': 0, 'nlg': speech}

        # 0-2) 상태 초기화 명령
        if self._is_reset_command(text_norm):
            self._reset_memory()
            speech = "현재 작업 상태와 도구 기억을 초기화했습니다."
            self._speak(speech)
            return {'mode': 'RESET', 'tool': 'NONE', 'section': 0, 'nlg': speech}

        # 0-3) 직전 작업 조회
        if self._is_last_task_query(text_norm):
            if self.last_task_desc:
                # last_task_desc를 자연어로 풀어서 TTS 생성
                desc = f"방금 수행한 작업은 {self.last_task_desc} 입니다."
                try:
                    speech = self.chat_engine.generate_speech(
                        task_description=desc,
                        style="친절하고 간결하게"
                    )
                except Exception:
                    speech = f"바로 직전에 {self.last_task_desc} 작업을 수행했습니다."
            else:
                speech = "아직 수행한 작업이 없습니다."
            self._speak(speech)
            return {'mode': 'INFO_LAST_TASK', 'tool': 'NONE', 'section': 0, 'nlg': speech}

        # 1) 외부 노드 모드는 미지원
        if self.use_external_nodes:
            self.get_logger().error("외부 노드 모드는 현재 지원되지 않습니다.")
            return {'mode':'ERROR', 'nlg':'외부 노드 모드 미지원'}

        # 2) 내장 LLM 호출
        try:
            parsed = self.chat_engine.parse_and_infer(raw_text)

            mode = parsed.get("mode", "NONE")
            tool_kr = parsed.get("tool_final", "NONE")
            section = parsed.get("section", 0)  # 이제 기본값 0
            tool_en = self._to_en_tool(tool_kr)

            # 2-1) 모드 없음 → 작업 명령이 명확하지 않음
            if not mode or mode == "NONE":
                speech = "작업 명령이 명확하지 않습니다. 시작, 전달, 반납, 종료, 정보, 조명 중에서 말씀해 주세요."
                self._speak(speech)
                return {'mode': 'NONE', 'tool': 'NONE', 'section': section, 'nlg': speech}

            # 2-2) 모드 유효성 검사
            if mode not in ALLOWED_MODES:
                error_msg = f"요청을 이해하지 못했습니다. (모드: {mode})"
                self._speak(error_msg)
                return {'mode':'ERROR', 'nlg':error_msg}

            # 2-3) DELIVER인데 도구가 없는 경우 → 공구 Clarification
            if mode == "DELIVER" and tool_en == "NONE":
                speech = "어떤 도구" \
                " 가져올까요? 니퍼, 커터, 와이어 스트리퍼 중에서 선택해주세요."
                self._speak(speech)
                return {'mode':'ASK_DELIVER_TOOL', 'tool':'NONE', 'section':section, 'nlg':speech}

            # 2-4) LIGHT_ON인데 section이 0인 경우 → 1번 섹션으로 보정
            light_on_auto_section_message = None
            if mode == "LIGHT_ON" and (not section or section == 0):
                light_on_auto_section_message = "지점이 명확하지 않아 기본 구역인 1번 섹션을 비추겠습니다."
                section = 1  # 실제 제어는 1번 섹션으로

            # 2-5) FINISH인데 사용 중 도구가 남아 있는 경우 → 반납+종료 여부 협의
            if mode == "FINISH" and len(self.active_tools) > 0:
                self.pending_finish_with_return = True
                tools_kr = ", ".join(_tool_kr_for_speech(t) for t in self.active_tools) if self.active_tools else "도구들"
                speech = f"현재 사용 중인 도구({tools_kr})가 남아 있습니다. 도구들을 모두 반납하고 종료할까요?"
                self._speak(speech)
                return {'mode': 'ASK_FINISH_WITH_RETURN', 'tool': 'NONE', 'section': section, 'nlg': speech}

            # 2-6) 이미 작업 중인데 또 작업 모드가 들어온 경우 (INFO는 예외)
            work_modes = {"START", "DELIVER", "RETURN", "FINISH", "LIGHT_ON", "LIGHT_OFF"}
            if self.active_mode is not None and mode in work_modes:
                current_desc = self._describe_current_task()
                desc = f"현재 {current_desc}이 진행 중입니다. 이전 작업이 끝난 뒤에 새 작업을 시작할 수 있습니다."
                try:
                    speech = self.chat_engine.generate_speech(
                        task_description=desc,
                        style="친절하고 간결하게"
                    )
                except Exception:
                    speech = "현재 다른 작업이 진행 중입니다. 작업이 끝난 뒤 다시 요청해 주세요."
                self._speak(speech)
                return {'mode': 'BUSY', 'tool': self.active_tool or "NONE", 'section': self.active_section, 'nlg': speech}

            # 3) 모드별 페이로드(Payload) 및 TTS 태스크 생성
            payload = {
                "mode": mode,
                "tool": tool_en,   # (INFO/DELIVER/RETURN 시 사용, 나머지는 "NONE" 가능)
                "section": section # (LIGHT_ON/LIGHT_OFF 시 사용, 나머지는 0 가능)
            }
            
            # TTS를 위한 작업 설명 문자열
            speech_task_desc = ""
            if mode == "INFO":
                speech_task_desc = f"INFO, 도구: {tool_kr}" if tool_kr != "NONE" else "INFO, 일반"
            elif mode == "LIGHT_ON":
                payload["tool"] = "NONE"
                speech_task_desc = f"LIGHT_ON, 구역: {section}"
            elif mode == "LIGHT_OFF":
                payload["tool"] = "NONE"
                speech_task_desc = f"LIGHT_OFF"
            elif mode == "DELIVER":
                speech_task_desc = f"DELIVER, 도구: {tool_kr}"
            elif mode == "RETURN":
                if tool_kr == "NONE":
                    speech_task_desc = "RETURN, 모든 도구 반납"
                else:
                    speech_task_desc = f"RETURN, 도구: {tool_kr}"
            elif mode == "START":
                payload["tool"] = "NONE"
                speech_task_desc = "START"
            elif mode == "FINISH":
                payload["tool"] = "NONE"
                speech_task_desc = "FINISH"

            # 4) 페이로드 발행
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.control_pub.publish(msg)
            self.get_logger().info(f"[CONTROL] {payload}")

            # 현재 실행 중 작업으로 기록 (INFO 제외)
            if mode in {"START", "DELIVER", "RETURN", "FINISH", "LIGHT_ON", "LIGHT_OFF"}:
                self.active_mode = mode
                self.active_tool = payload.get("tool", "NONE")
                self.active_section = payload.get("section", 0)

            # 5) TTS 음성 생성
            try:
                # LIGHT_ON에서 자동 보정 메시지가 있으면 그 설명을 사용
                if light_on_auto_section_message and mode == "LIGHT_ON":
                    speech = light_on_auto_section_message
                else:
                    speech = self.chat_engine.generate_speech(
                        task_description=speech_task_desc,
                        style="친절하고 간결하게"
                    )
            except Exception as e:
                self.get_logger().error(f"NLG (generate_speech) 실패: {e}")
                speech = "알겠습니다. 지시에 따르겠습니다."

            self._speak(speech)

            return {'mode':mode,'tool':tool_en,'section':section, 'nlg':speech}

        except Exception as e:
            self.get_logger().error(f"process_chat_message 오류: {e}")
            import traceback; traceback.print_exc()
            error_msg = '죄송합니다. 요청을 처리하는 중 오류가 발생했습니다.'
            self._speak(error_msg)
            return {'mode':'ERROR','tool':'NONE','section':0, 'nlg':error_msg}

# ----- 전역 및 가드 -----
ros2_node: Optional[WebUINode] = None
ros2_executor: Optional[MultiThreadedExecutor] = None
ros2_thread: Optional[threading.Thread] = None
_ros_inited = False
_shutting_down = False

def init_ros2(use_external_nodes=False):
    global ros2_node, ros2_executor, ros2_thread, _ros_inited
    if _ros_inited: return True
    try:
        print(f"[ROS2] 초기화 중... (외부 노드 모드: {use_external_nodes})")
        rclpy.init()
        _ros_inited = True
        ros2_node = WebUINode(use_external_nodes=use_external_nodes)
        ros2_executor = MultiThreadedExecutor(); ros2_executor.add_node(ros2_node)
        def run_ros2():
            try: ros2_executor.spin()
            except Exception as e: print(f"[ROS2] 실행 중 오류: {e}")
        ros2_thread = threading.Thread(target=run_ros2, daemon=True); ros2_thread.start()
        print("[ROS2] 초기화 완료"); return True
    except Exception as e:
        print(f"[ROS2] 초기화 실패: {e}")
        import traceback; traceback.print_exc()
        _ros_inited = False
        return False

def shutdown_ros2():
    global ros2_node, ros2_executor, ros2_thread, _ros_inited, _shutting_down
    if _shutting_down: return
    _shutting_down = True
    try:
        if ros2_executor:
            try: ros2_executor.shutdown()
            except: pass
        if ros2_node:
            try: ros2_node.destroy_node()
            except: pass
        if _ros_inited and rclpy.ok():
            try: rclpy.shutdown()
            except: pass
        if ros2_thread and ros2_thread.is_alive():
            ros2_thread.join(timeout=1.0)
        print("[ROS2] 정리 완료")
    finally:
        ros2_node = None; ros2_executor = None; ros2_thread = None
        _ros_inited = False; _shutting_down = False

# ----- Flask 라우트 -----
@app.route("/")
def index(): return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get('message', '') if data else ''
    if not ros2_node: return jsonify({'reply': "ROS2 노드가 초기화되지 않았습니다."}), 500
    try:
        result = ros2_node.process_chat_message(user_input)
        return jsonify({'reply': result.get('nlg', '응답을 생성할 수 없습니다.'),'control': result})
    except Exception:
        return jsonify({'reply': "메시지 처리 오류가 발생했습니다."}), 500

# 선택: 외부에서 모드 전환 사용 시
@app.route("/mode", methods=["POST"])
def switch_mode():
    data = request.get_json()
    use_external = data.get('use_external_nodes', False)
    try:
        shutdown_ros2()
        if init_ros2(use_external_nodes=use_external):
            return jsonify({'success': True,'mode': ("외부 노드" if use_external else "내장"),
                            'use_external_nodes': use_external})
        else:
            return jsonify({'success': False, 'error': 'ROS2 재초기화 실패'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ----- SocketIO -----
@socketio.on('connect')
def handle_connect():
    emit('system_status', {'ros2_initialized': ros2_node is not None,
                           'ros2_running': ros2_thread is not None and ros2_thread.is_alive(),
                           'timestamp': time.time()})
    if ros2_node and hasattr(ros2_node, 'last_tool_info'):
        emit('tool_detection', ros2_node.last_tool_info)

@socketio.on('disconnect')
def handle_disconnect(): pass

# ----- 엔트리포인트 -----
def main():
    print("[서버] ROS2 통합 Flask-SocketIO 서버를 시작합니다.")
    if not init_ros2(use_external_nodes=False):
        print("[오류] ROS2 초기화 실패"); sys.exit(1)
    try:
        socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_ros2()

if __name__ == "__main__":
    main()
