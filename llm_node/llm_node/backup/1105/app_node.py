# -*- coding: utf-8 -*-
"""
app_node.py
- Flask-SocketIO 웹 UI 서버
- ROS2 통합
- llm_inference (v5.11) 초간단 버전에 호환됨
- 'mode', 'tool_final', 'section' 3개 키만 사용
- 'FOCUS' 모드 지원
- 'generate_speech' 함수 호출
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
from llm_inference import ToolChatEngine

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

# ----- ➖ 길이 파싱 (제거됨) -----

# ----- 도구/모드 -----
NIPPER_SYNS = ["니퍼","니뻐","니빠","롱노즈","롱노즈 플라이어","롱노우즈 플라이어","롱 플라이어","롱플라이어",
               "needle nose","needle-nose","long nose","long-nose","nipper"]
TOOL_EN_MAP = {**{k: "nipper" for k in NIPPER_SYNS},
               "커터":"wire_cutter","와이어 스트리퍼":"wire_stripper","M3 나사":"M3","스탠드":"lamp",
               "wire_cutter":"wire_cutter","wire_stripper":"wire_stripper","M3":"M3","lamp":"lamp","NONE":"NONE"}
TOOL_KR_MAP = {"nipper":"니퍼","wire_cutter":"커터","wire_stripper":"와이어 스트리퍼","M3":"M3 나사","lamp":"스탠드","NONE":""}

# ALLOWED_MODES 6개로 축소
ALLOWED_MODES = {"START","DELIVER","RETURN","FINISH","INFO","FOCUS"}

# 방향(Direction) 관련 딕셔너리 제거
# _delivery_speech 함수 제거

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
        # ➖ pub_length (길이) 제거
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

    def task_done_callback(self, msg: String):
        text = msg.data or ""
        socketio.emit('task_done', {'text': text, 'timestamp': time.time()})

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

    # ➖ _map_control_with_tool (단순화된 페이로드 사용으로 제거)

    # ➕ process_chat_message (v5.11 호환 단순화)
    def process_chat_message(self, user_input: str) -> Dict[str, Any]:
        
        # 1) ➖ 길이 파싱 로직 제거
        
        # 2) ➖ 외부 노드 로직 제거 (내장 LLM만 사용)
        if self.use_external_nodes:
            self.get_logger().error("외부 노드 모드는 현재 지원되지 않습니다.")
            return {'mode':'ERROR', 'nlg':'외부 노드 모드 미지원'}

        # 3) 내장 LLM (v5.11) 호출
        try:
            # llm_inference은 3개의 키만 반환:
            # {"mode", "tool_final", "section"}
            parsed = self.chat_engine.parse_and_infer(user_input)

            mode = parsed.get("mode", "NONE")
            tool_kr = parsed.get("tool_final", "NONE")
            section = parsed.get("section", 1) # 기본값 1로 설정됨
            
            tool_en = self._to_en_tool(tool_kr)

            # 3-1) ask_back, plan_action, should_call 로직 제거
            
            # 3-2) 모드 유효성 검사
            if mode not in ALLOWED_MODES:
                error_msg = f"요청을 이해하지 못했습니다. (모드: {mode})"
                if hasattr(self, "chat_out_pub"):
                    m = String(); m.data = error_msg; self.chat_out_pub.publish(m)
                return {'mode':'ERROR', 'nlg':error_msg}

            # 3-3) ➕ DELIVER 모드 공구 필수 확인 (v5.11에 ask_back이 없으므로 app에서 처리)
            if mode == "DELIVER" and tool_en == "NONE":
                speech = "어떤 공구가 필요하신가요?"
                if hasattr(self, "chat_out_pub"):
                    m = String(); m.data = speech; self.chat_out_pub.publish(m)
                # ℹ️ 페이로드 발행 없이 종료
                return {'mode':'ASK_BACK', 'tool':'NONE', 'section':section, 'nlg':speech}

            # 3-4) ➕ 모드별 페이로드(Payload) 및 TTS 태스크 생성 (단순화)
            payload = {
                "mode": mode,
                "tool": tool_en, # (INFO/DELIVER/RETURN 시 사용, 나머지는 "NONE")
                "section": section # (FOCUS 시 사용, 나머지는 기본값 1)
            }
            
            # TTS를 위한 작업 설명 문자열
            speech_task_desc = ""
            if mode == "INFO":
                speech_task_desc = f"INFO, 도구: {tool_kr}" if tool_kr != "NONE" else "INFO, 일반"
            elif mode == "FOCUS":
                # tool_en은 무시하고 section만 사용
                payload["tool"] = "NONE" 
                speech_task_desc = f"FOCUS, 구역: {section}"
            elif mode == "DELIVER":
                speech_task_desc = f"DELIVER, 도구: {tool_kr}"
            elif mode == "RETURN":
                speech_task_desc = f"RETURN, 도구: {tool_kr}" if tool_kr != "NONE" else "RETURN, 전체 반납"
            elif mode == "START":
                payload["tool"] = "NONE"
                speech_task_desc = "START"
            elif mode == "FINISH":
                payload["tool"] = "NONE"
                speech_task_desc = "FINISH"

            # 4. 페이로드 발행
            msg = String(); msg.data = json.dumps(payload, ensure_ascii=False)
            self.control_pub.publish(msg)
            self.get_logger().info(f"[CONTROL] {payload}")

            # 5. ➕ TTS 음성 생성 (generate_speech 호출)
            try:
                speech = self.chat_engine.generate_speech(task_description=speech_task_desc, style="친절하고 간결하게")
            except Exception as e:
                self.get_logger().error(f"NLG (generate_speech) 실패: {e}")
                speech = "알겠습니다. 지시에 따르겠습니다."

            if hasattr(self, "chat_out_pub"):
                m = String(); m.data = speech; self.chat_out_pub.publish(m)

            return {'mode':mode,'tool':tool_en,'section':section, 'nlg':speech}

        except Exception as e:
            self.get_logger().error(f"process_chat_message 오류: {e}")
            import traceback; traceback.print_exc()
            error_msg = '죄송합니다. 요청을 처리하는 중 오류가 발생했습니다.'
            if hasattr(self, "chat_out_pub"):
                m = String(); m.data = error_msg; self.chat_out_pub.publish(m)
            return {'mode':'ERROR','tool':'NONE','section':1, 'nlg':error_msg}

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