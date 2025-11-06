#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

ALLOWED_TOOLS = {"nipper", "vernier_calipers", "wire_cutter", "wire_stripper"}

class ControlCommandsListener(Node):
    def __init__(self):
        super().__init__('control_commands_listener')
        self.sub_llm = self.create_subscription(String, '/control_commands', self.on_cmd, 10)
        self.pub_tool = self.create_publisher(String, '/info/string/obj_name', 10)
        self.reset()

    def reset(self):
        self.tool = None
        self.mode = None
        self.direction = None
        self.target = None
        self.control = None
        self.command = None

    def on_cmd(self, msg: String):
        self.get_logger().info(f"[RAW] {msg.data}")
        try:
            obj = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"JSON 파싱 실패: {e}")
            return

        # 상태 업데이트
        self.tool      = obj.get('tool')
        self.mode      = obj.get('mode')
        self.direction = obj.get('direction')
        self.target    = obj.get('target')
        self.control   = obj.get('control')

        self.get_logger().info(
            f"[PARSED] mode={self.mode}, tool={self.tool}, "
            f"direction={self.direction}, target={self.target}, control={self.control}"
        )

        # 명령 결정
        if self.mode == 'DELIVER':
            if self.direction not in (None, "NONE"):
                self.get_logger().info("시나리오: [DELIVER with direction]")
                self.command = 'DELIVER_0'
            
            else:
                if self.tool == 'M3':
                    self.get_logger().info("시나리오: M3 나사 가져다주기")
                    self.command = 'DELIVER_1'
                else:
                    self.get_logger().info("시나리오: 일반 도구 가져다주기")
                    self.command = 'DELIVER_2'
        elif self.mode == 'RETURN':
            self.command = 'RETURN'
        
        elif self.mode == 'UP':
            self.command = 'UP'
        
        elif self.mode == 'DOWN':
            self.command = 'DOWN'
        
        elif self.mode == 'DISASSEMBLE':
            self.command = 'DISASSEMBLE'
        
        elif self.mode == 'START':
            self.command = 'START'
        
        elif self.mode == 'FINISH':
            self.command = 'FINISH'
        
        else:
            self.get_logger().info("모드를 인식하지 못했습니다.")
            self.command = None

        # 즉시 처리
        self.process_command()

    def tool_publish(self, tool: str):
        msg = String()
        msg.data = tool
        self.pub_tool.publish(msg)
        self.get_logger().info(f"{tool} publish 완료")

    def process_command(self):
        if self.command is None:
            return
        try:
            if self.command == "DELIVER_2" or self.command == "DELIVER_0" or self.command == "DELIVER_1":
                if self.tool in ALLOWED_TOOLS:
                    self.tool_publish(self.tool)
                else:
                    self.get_logger().info("도구가 허용 목록에 없거나 탐지되지 않았습니다")
            elif self.command in {"DELIVER_0","DELIVER_1","RETURN","UP","DOWN","DISASSEMBLE","START","FINISH"}:
                self.get_logger().info(f"[{self.command}] 기능 구축 중입니다")
            else:
                self.get_logger().info("정확한 명령이 전송되지 않았습니다")
        finally:
            # 중복 실행 방지
            self.command = None

def main(args=None):
    rclpy.init(args=args)
    node = ControlCommandsListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
