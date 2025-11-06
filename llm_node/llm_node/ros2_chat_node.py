# -*- coding: utf-8 -*-

import json, os, sys, warnings, re
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# 동일 디렉터리 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from llm_inference import ToolChatEngine

# ===================== 숫자 파싱 유틸 (cm 전용) =====================
_CM_UNITS = r'(?:cm|센치|센티|센티미터|센치미터)'

QOS_LATCED = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,  # latched 유사
)
QOS_RELIABLE = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


def _to_float_num(s: str) -> float:
    # "1,200.5" → 1200.5
    return float(s.replace(",", ""))

def parse_length_cm(text: str):
    """
    입력 문자열 끝에서 cm 단위 길이만 파싱하여 float(cm) 반환.
    지원 예: "16", "16cm", "3.5 cm", "10.6센치", "10 점 6 센티", "10~12cm", "10-12센치"
    범위는 평균값 사용.
    """
    if text is None:
        return None
    # 정규화: 공백 제거, '점'을 소수점으로, 소문자
    t = str(text).lower().replace(" ", "").replace("점", ".")
    # 1) 범위: 10~12cm, 10-12센치
    m = re.search(rf'(-?\d+(?:[.,]\d+)?)\s*(?:~|-)\s*(-?\d+(?:[.,]\d+)?)(?:\s*{_CM_UNITS})?$', t)
    if m:
        a, b = m.groups()
        return (_to_float_num(a) + _to_float_num(b)) / 2.0  # cm
    # 2) 단일 수치(+선택적 단위): 10.6센치, 16, 3.5cm
    m = re.search(rf'(-?\d+(?:[.,]\d+)?)(?:\s*{_CM_UNITS})?$', t)
    if m:
        return _to_float_num(m.group(1))  # cm
    return None
# ================================================================
NIPPER_SYNS = [
    "니퍼","니뻐","니빠",
    "롱노즈","롱노즈 플라이어","롱노우즈 플라이어","롱 플라이어","롱플라이어",
    "needle nose","needle-nose","long nose","long-nose","nipper"
]
TOOL_EN_MAP = {
    # 니퍼 계열 전부 nipper로
    **{k: "nipper" for k in NIPPER_SYNS},

    # 그 외
    "커터": "wire_cutter",
    "와이어 스트리퍼": "wire_stripper",
    "M3 나사": "M3",
    "스탠드": "lamp",

    # 영문 그대로 허용
    "wire_cutter": "wire_cutter",
    "wire_stripper": "wire_stripper",
    "M3": "M3",
    "lamp": "lamp",
    "NONE": "NONE",
}

TOOL_KR_MAP = {
    "nipper": "니퍼",
    "wire_cutter": "커터",
    "wire_stripper": "와이어 스트리퍼",
    "M3": "M3 나사",
    "lamp": "스탠드",
    "NONE": "",
}

MODE_HELP = "가능한 Mode: START(시작) | UP(스탠드 올림) | DOWN(스탠드 내림) | DELIVER(가져다주기) | RETURN(반납) | DISASSEMBLE(분해) | MEASURE(검류 측정) | FINISH(종료)"
ALLOWED_MODES = {"START", "UP", "DOWN", "DELIVER", "RETURN", "DISASSEMBLE", "MEASURE", "FINISH"}

DIR_KR = {"left": "왼쪽", "right": "오른쪽", "front": "앞쪽", "back": "뒤쪽", "center": "정중앙", "NONE": ""}

def _tool_kr_for_speech(tool_en: str) -> str:
    kr = TOOL_KR_MAP.get(tool_en, tool_en or "")
    if kr.upper().startswith("M3"):
        return "엠쓰리 나사"
    return kr

# 받침 판단 → 을/를
def _has_batchim(k: str) -> bool:
    if not k:
        return False
    ch = k.strip()[-1]
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return False
    return (code % 28) != 0

def _eulreul(word: str) -> str:
    return "을" if _has_batchim(word) else "를"

# DELIVER 전용 규칙식 발화
def _delivery_speech(tool_en: str, target_en: str, direction: str) -> str:
    tool_k = _tool_kr_for_speech(tool_en)
    target_k = _tool_kr_for_speech(target_en) if target_en != "NONE" else ""
    dir_k = DIR_KR.get(direction, "")
    if target_en != "NONE" and direction != "NONE":
        return f"네, {tool_k}{_eulreul(tool_k)} {target_k} {dir_k}에 가져다 드릴게요."
    if target_en != "NONE":
        return f"네, {tool_k}{_eulreul(tool_k)} {target_k}에 가져다 드릴게요."
    if direction != "NONE":
        return f"네, {tool_k}{_eulreul(tool_k)} {dir_k}에 가져다 드릴게요."
    return f"네, {tool_k}{_eulreul(tool_k)} 바로 가져다 드릴게요."

class ToolChatNode(Node):
    def __init__(self):
        super().__init__('tool_chat_node')

        self.declare_parameter('model_path', '/home/liw/sllm/models/gemma3-q4_k_m_budda.gguf')
        self.declare_parameter('mode_rag_jsonl_path', '/home/temp_id/ros2_ws/src/vision_node/rule.jsonl')
        self.declare_parameter('faiss_index_path', './faiss_index_ros')
        self.declare_parameter('use_gpu_llama', True)
        self.declare_parameter('engine_verbose', True)

        model_path       = self.get_parameter('model_path').get_parameter_value().string_value
        mode_rag_jsonl   = self.get_parameter('mode_rag_jsonl_path').get_parameter_value().string_value
        faiss_index_path = self.get_parameter('faiss_index_path').get_parameter_value().string_value
        use_gpu_llama    = self.get_parameter('use_gpu_llama').get_parameter_value().bool_value
        engine_verbose   = self.get_parameter('engine_verbose').get_parameter_value().bool_value

        self.sub_in  = self.create_subscription(String, '/tool_chat/in',  self.on_text, QOS_LATCED)
        self.pub_cmd = self.create_publisher(String, '/zeus/string/llm_cmd',  QOS_RELIABLE)
        self.pub_out = self.create_publisher(String, '/tool_chat/out',  QOS_RELIABLE)
        self.pub_length = self.create_publisher(Float32, '/zeus/float/length_val', QOS_LATCED)

        self.get_logger().info("ToolChatNode initializing...")

        try:
            self.engine = ToolChatEngine(
                model_path=model_path,
                mode_rag_jsonl_path=mode_rag_jsonl,
                faiss_index_path=faiss_index_path,
                use_gpu_llama=use_gpu_llama,
                verbose=engine_verbose
            )
            assert hasattr(self.engine, "generate_speech")
        except Exception as e:
            self.get_logger().error(f"ToolChatEngine 초기화 실패: {e}")
            rclpy.shutdown()
            sys.exit(1)

        if not os.path.isfile(mode_rag_jsonl):
            pass

        self.get_logger().info("ToolChatNode is up. Sub:/tool_chat/in  Pub:/zeus/string/llm_cmd, /tool_chat/out, /zeus/float/length_val")

    def _to_en_tool(self, tool: str) -> str:
        if not tool:
            return "NONE"
        tool = tool.strip()
        return TOOL_EN_MAP.get(tool, tool)

    def _map_control_with_tool(self, control: str, mode: str, tool_en: str, direction: str, target_en: str) -> str:
        if mode not in ("DELIVER", "RETURN"):
            return control or (mode or "")
        if not control:
            base = f"{tool_en}+{mode}" if tool_en != "NONE" else mode
            if mode == "DELIVER":
                if target_en and target_en != "NONE":
                    base += f"+{target_en}"
                if direction and direction != "NONE":
                    base += f"+{direction}"
            return base
        parts = control.split("+")
        if len(parts) >= 2:
            parts[0] = tool_en or parts[0]
            if mode == "DELIVER":
                has_target = len(parts) >= 3 and parts[2] in TOOL_EN_MAP.values()
                if target_en and target_en != "NONE":
                    if len(parts) < 3:
                        parts.append(target_en)
                    elif not has_target:
                        parts[2] = target_en
                if direction and direction != "NONE":
                    if len(parts) < 4:
                        parts.append(direction)
                    else:
                        if parts[-1] not in ("left", "right", "front", "back", "center"):
                            parts.append(direction)
            return "+".join(parts)
        return control

    def on_text(self, msg: String):
        user_text = (msg.data or "").strip()
        if not user_text:
            return

        # 1) cm 길이 우선 파싱 및 퍼블리시
        val_cm = parse_length_cm(user_text)
        if val_cm is not None:
            out = Float32()
            out.data = float(val_cm)  # cm
            self.pub_length.publish(out)
            self.get_logger().info(f"[LENGTH_CM] {out.data:.3f} cm")
            # 길이 테스트 전용이면 조기 종료
            return

        # 2) 이하 기존 LLM 기반 플로우
        self.get_logger().info(f"[IN] {user_text}")

        parsed = self.engine.parse_and_infer(user_text)
        plan = self.engine.plan_action(parsed, raw_text=user_text)

        if plan.get("ask_back") and plan.get("confirm_text"):
            self._publish_out(plan["confirm_text"])
            return

        if plan.get("should_call_function"):
            mode        = plan.get("mode", "NONE")
            tool_kr     = plan.get("tool_final", "NONE")
            target_kr   = plan.get("target_object", "NONE")
            direction   = plan.get("direction") or "NONE"
            control     = plan.get("control_string") or ""

            if mode not in ALLOWED_MODES:
                self._publish_out(MODE_HELP)
                return

            tool_en   = self._to_en_tool(tool_kr)
            target_en = self._to_en_tool(target_kr)
            dir_kr    = DIR_KR.get(direction, "")

            control_mapped = self._map_control_with_tool(control, mode, tool_en, direction, target_en)

            # 제어 페이로드의 방향 기본값
            if mode == 'DELIVER':
                if direction not in (None, "NONE"):
                    payload_direction = direction
                else:
                    payload_direction = 'straight' if tool_en == 'M3' else 'down'
            elif mode == 'RETURN':
                tool_en == 'NONE'
                target_en == 'NONE'
                payload_direction = 'down'
            elif mode in ('UP', 'DOWN'):
                payload_direction = 'straight'
            else:
                payload_direction = 'NONE'

            payload = {
                "mode": mode,
                "tool": tool_en if mode in ("DELIVER", "RETURN") else "NONE",
                "direction": payload_direction if mode == "DELIVER" else "NONE",
                "target": target_en if mode == "DELIVER" and target_en != "NONE" else "NONE",
                "control": control_mapped,
            }

            self._publish_cmd(payload)
            self.get_logger().info(f"[CONTROL] {control_mapped}")

            # NLG: DELIVER는 규칙식, 그 외 LLM
            try:
                if mode == "DELIVER" and tool_en != "NONE":
                    speech = _delivery_speech(tool_en, target_en, direction)
                    self._publish_out(speech)
                    return

                intent_details = f"모드: {mode}"
                if mode in ("UP", "DOWN"):
                    intent_details += ", 도구: 스탠드"
                if tool_en != "NONE" and mode in ("DELIVER"):
                    intent_details += f", 도구: {_tool_kr_for_speech(tool_en)}"
                if target_en != "NONE" and mode == "DELIVER":
                    intent_details += f", 배치 대상: {_tool_kr_for_speech(target_en)}"
                if direction != "NONE" and mode == "DELIVER":
                    intent_details += f", 방향: {dir_kr}"

                speech = self.engine.generate_speech(
                    task_description=intent_details,
                    style="친절하고 간결하게",
                )
            except Exception as e:
                self.get_logger().warning(f"LLM 문장 생성 실패(폴백 사용): {e}")
                speech = "알겠습니다. 지시에 따르겠습니다."

            self._publish_out(speech)
        else:
            mode = plan.get("mode", "NONE")
            if mode != "NONE":
                self._publish_out("모드는 인식했지만 작업에 필요한 정보가 더 필요합니다.")
            else:
                self._publish_out("요청을 이해하지 못했습니다. 다시 한번 말씀해 주세요.")

    def _publish_cmd(self, obj: dict):
        msg = String()
        msg.data = json.dumps(obj, ensure_ascii=False)
        self.pub_cmd.publish(msg)
        self.get_logger().info(f"[CMD] {msg.data}")

    def _publish_out(self, text: str):
        msg = String()
        msg.data = text
        self.pub_out.publish(msg)
        self.get_logger().info(f"[OUT] {text}")

def main(args=None):
    rclpy.init(args=args)
    node = ToolChatNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
