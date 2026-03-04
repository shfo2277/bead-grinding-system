#!/usr/bin/env python3
"""
PA_grinding TCP 클라이언트 + 오케스트레이터 (ROS2)
- 로봇 TCP 제어 (4 소켓)
- 노드 자동 실행/종료 (subprocess)
- 자동 그라인딩 루프: 스캔 → 인식 → 경로 → 가공 → 판정 → 반복
"""

import os
import socket
import struct
import time
import signal
import threading
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Path
from std_msgs.msg import Bool

# ==============================
# 설정
# ==============================
#HOST = '172.31.0.1'
HOST = '192.168.1.254'


PORT_MOTION = 11000
PORT_COMMAND = 11001
PORT_FEEDBACK = 11002
PORT_JOINT = 11003

WORKSPACE = '/workspace/BEADtrain'
CONTOUR_CSV = os.path.join(WORKSPACE, 'bead_contour.csv')
WIDTH_CSV = os.path.join(WORKSPACE, 'bead_width.csv')

STATUS_CODES = {
    0: "이동 중",
    1: "홈 위치",
    2: "스캔 위치 도달",
    3: "가공 준비 완료",
    4: "가공 완료"
}

CMD_DONE_STATUS = {
    1: [1],
    2: [2],
    3: [3],
    4: [4],
}

last_status = -1


# ==============================
# ROS2 수신 노드
# ==============================
class ReceiverNode(Node):
    def __init__(self):
        super().__init__('grind_receiver')
        self.waypoints = []
        self.path_received = False
        self.stop_signal = False

        sensor_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
        )

        self.sub_path = self.create_subscription(
            Path, '/bead/grind_path_base', self.path_callback, sensor_qos)

        self.sub_stop = self.create_subscription(
            Bool, '/stop', self.stop_callback, 10)

        self.get_logger().info('ReceiverNode ready.')

    def path_callback(self, msg: Path):
        self.waypoints = []
        for pose_st in msg.poses:
            self.waypoints.append((
                pose_st.pose.position.x * 1000.0,
                pose_st.pose.position.y * 1000.0,
                pose_st.pose.position.z * 1000.0,
            ))
        if not self.path_received and self.waypoints:
            p0, pn = self.waypoints[0], self.waypoints[-1]
            self.get_logger().info(
                f'Path: {len(self.waypoints)} pts, '
                f'Start({p0[0]:.1f},{p0[1]:.1f},{p0[2]:.1f}) '
                f'End({pn[0]:.1f},{pn[1]:.1f},{pn[2]:.1f}) mm')
        self.path_received = True

    def stop_callback(self, msg: Bool):
        if msg.data:
            self.stop_signal = True
            self.get_logger().info('STOP signal received!')

    def reset(self):
        self.waypoints = []
        self.path_received = False
        self.stop_signal = False


def spin_ros2(node):
    rclpy.spin(node)


# ==============================
# 서브프로세스 관리
# ==============================
def ensure_align_depth():
    """aligned_depth_to_color 토픽 활성화 확인/설정"""
    cmd = (
        "source /opt/ros/humble/setup.bash && "
        "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && "
        "ros2 param set /camera/camera align_depth.enable true"
    )
    try:
        result = subprocess.run(
            ['bash', '-c', cmd],
            capture_output=True, text=True, timeout=10
        )
        if 'successful' in result.stdout:
            print("  [ALIGN] align_depth.enable = true 확인")
            return True
        else:
            print(f"  [ALIGN] 설정 실패: {result.stdout.strip()} {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"  [ALIGN] 오류: {e}")
        return False


def start_node(script_name):
    """BEADtrain 폴더에서 Python 노드 실행"""
    cmd = (
        f"source /opt/ros/humble/setup.bash && "
        f"export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && "
        f"cd {WORKSPACE} && python3 -u {script_name}"
    )
    proc = subprocess.Popen(
        ['bash', '-c', cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    print(f"  [NODE] {script_name} started (PID {proc.pid})")
    return proc


def stop_node(proc, name=""):
    """노드 프로세스 종료"""
    if proc and proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        print(f"  [NODE] {name} stopped")


def wait_for_csv(timeout=30, old_mtime=None):
    """CSV 파일이 생성/갱신될 때까지 대기"""
    if old_mtime is None:
        old_mtime = 0
        if os.path.exists(CONTOUR_CSV):
            old_mtime = os.path.getmtime(CONTOUR_CSV)

    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(CONTOUR_CSV) and os.path.exists(WIDTH_CSV):
            new_mtime = os.path.getmtime(CONTOUR_CSV)
            if new_mtime > old_mtime:
                print(f"  CSV 저장 확인 (contour + width)")
                return True
        time.sleep(0.5)

    print(f"  [경고] CSV 저장 대기 타임아웃 ({timeout}초)")
    return False


# ==============================
# TCP 함수들
# ==============================
def connect_sockets():
    sockets = {}
    for name, port in [('Command', PORT_COMMAND), ('Motion', PORT_MOTION),
                        ('Feedback', PORT_FEEDBACK), ('Joint', PORT_JOINT)]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((HOST, port))
            s.settimeout(1.0)
            sockets[name] = s
            print(f"[연결] {name} ({port}) 성공")
        except Exception as e:
            print(f"[실패] {name}: {e}")
            sockets[name] = None
    return sockets


def send_command(sock, cmd):
    if sock is None:
        print("[오류] Command 소켓 없음")
        return False
    sock.send(bytes([cmd]))
    print(f"[송신] CMD = {cmd}")
    return True


def wait_feedback(sock, done_statuses, timeout=60):
    global last_status
    if sock is None:
        print("[오류] Feedback 소켓 없음")
        return None

    print(f"[대기] Feedback 수신 대기 중...")
    sock.settimeout(3.0)
    start = time.time()

    while time.time() - start < timeout:
        try:
            data = sock.recv(1024)
            if data:
                for byte in data:
                    if byte <= 4:
                        status = byte
                    elif 48 <= byte <= 52:
                        status = byte - 48
                    else:
                        status = byte

                    status_text = STATUS_CODES.get(status, f"Unknown({status})")
                    print(f"  ★ STATUS = {status} : {status_text}")
                    last_status = status

                    if status in done_statuses:
                        print(f"  ✔ 완료!")
                        return status
        except socket.timeout:
            elapsed = int(time.time() - start)
            print(f"  ... 대기 중 ({elapsed}초)")
            continue
        except Exception as e:
            print(f"[오류] {e}")
            return None

    print(f"[타임아웃] {timeout}초 경과")
    return last_status


def send_waypoints(sock, waypoints):
    if sock is None:
        print("[오류] Motion 소켓 없음")
        return False

    count = len(waypoints)
    data = struct.pack('<f', float(count))
    for p in waypoints:
        data += struct.pack('<fff', *p)

    sock.send(data)
    print(f"[송신] Waypoints: {count}개, {len(data)} bytes")
    for i, p in enumerate(waypoints):
        print(f"  P[{i}] = X:{p[0]:.1f}, Y:{p[1]:.1f}, Z:{p[2]:.1f}")
    return True


def close_sockets(sockets):
    for name, sock in sockets.items():
        if sock:
            sock.close()
            print(f"[종료] {name} 소켓 닫힘")


def status_text():
    if last_status == -1:
        return "초기"
    return STATUS_CODES.get(last_status, f"Unknown({last_status})")


# ==============================
# 자동 그라인딩 루프
# ==============================
def auto_grind_loop(sockets, ros_node):
    cycle = 0

    while True:
        cycle += 1
        print()
        print("=" * 50)
        print(f"  [AUTO] ===== 사이클 {cycle} =====")
        print("=" * 50)

        # ── STEP 1: 스캔 위치 이동 (CMD 2) ──
        print(f"\n[STEP 1] 스캔 위치 이동 (CMD 2)")
        send_command(sockets['Command'], 2)
        status = wait_feedback(sockets['Feedback'], CMD_DONE_STATUS[2])
        if status not in CMD_DONE_STATUS[2]:
            print("[오류] 스캔 위치 이동 실패! 중단.")
            return

        # ── STEP 2: 비드 인식 (bead_unet_node) ──
        print(f"\n[STEP 2] 비드 인식 시작 (bead_unet_node)")
        ros_node.reset()
        # CSV mtime을 subprocess 시작 전에 기록
        old_mtime = 0
        if os.path.exists(CONTOUR_CSV):
            old_mtime = os.path.getmtime(CONTOUR_CSV)
        proc_unet = start_node('bead_unet_node.py')
        time.sleep(3)  # 모델 로딩 대기

        csv_ok = wait_for_csv(timeout=30, old_mtime=old_mtime)

        if not csv_ok:
            stop_node(proc_unet, 'bead_unet_node')
            print("[오류] 비드 인식 실패! 중단.")
            return
        print("  비드 외곽선 + 폭 CSV 저장 완료")

        # ── STEP 3: 경로 생성 (bead_pose_node) ──
        # bead_unet_node가 /bead/mask를 계속 발행해야 하므로 아직 종료하지 않음
        print(f"\n[STEP 3] 3D 경로 생성 (bead_pose_node)")
        ensure_align_depth()
        time.sleep(1)  # aligned depth 토픽 안정화 대기
        proc_pose = start_node('bead_pose_node.py')

        # 경로 수신 대기
        path_timeout = 30
        start_t = time.time()
        while time.time() - start_t < path_timeout:
            if ros_node.path_received and ros_node.waypoints:
                break
            time.sleep(0.5)

        stop_node(proc_unet, 'bead_unet_node')
        stop_node(proc_pose, 'bead_pose_node')

        if not ros_node.waypoints:
            print("[오류] 경로 생성 실패! 중단.")
            return
        print(f"  경로: {len(ros_node.waypoints)} waypoints (mm)")

        # ── STEP 4: 경로 전송 + 가공 준비 (CMD 3) ──
        print(f"\n[STEP 4] 경로 전송 + 가공 준비 (CMD 3)")
        send_command(sockets['Command'], 3)
        time.sleep(0.1)
        send_waypoints(sockets['Motion'], ros_node.waypoints)

        status = wait_feedback(sockets['Feedback'], CMD_DONE_STATUS[3])
        if status not in CMD_DONE_STATUS[3]:
            print("[오류] 가공 준비 실패! 중단.")
            return

        # ── STEP 5: 가공 시작 (CMD 4) ──
        print(f"\n[STEP 5] 가공 시작 (CMD 4)")
        send_command(sockets['Command'], 4)
        status = wait_feedback(sockets['Feedback'], CMD_DONE_STATUS[4], timeout=120)
        if status not in CMD_DONE_STATUS[4]:
            print("[오류] 가공 완료 대기 실패! 중단.")
            return

        # ── STEP 6: 스캔 복귀 (CMD 2) ──
        print(f"\n[STEP 6] 스캔 위치 복귀 (CMD 2)")
        send_command(sockets['Command'], 2)
        status = wait_feedback(sockets['Feedback'], CMD_DONE_STATUS[2])
        if status not in CMD_DONE_STATUS[2]:
            print("[오류] 스캔 복귀 실패! 중단.")
            return

        # ── STEP 7: 종료 판정 (bead_unet_node_after) ──
        print(f"\n[STEP 7] 그라인딩 판정 (bead_unet_node_after)")
        ros_node.stop_signal = False
        proc_after = start_node('bead_unet_node_after.py')
        time.sleep(3)  # 모델 로딩 대기

        stop_timeout = 30
        start_t = time.time()
        while time.time() - start_t < stop_timeout:
            if ros_node.stop_signal:
                break
            time.sleep(0.5)

        stop_node(proc_after, 'bead_unet_node_after')

        if ros_node.stop_signal:
            print()
            print("=" * 50)
            print(f"  그라인딩 완료! (총 {cycle} 사이클)")
            print("=" * 50)
            return
        else:
            print(f"  비드 잔여 → 재가공 필요 (사이클 {cycle} 종료)")
            print("  → 다음 사이클...")


# ==============================
# 메인
# ==============================
def main():
    rclpy.init()
    ros_node = ReceiverNode()
    ros_thread = threading.Thread(target=spin_ros2, args=(ros_node,), daemon=True)
    ros_thread.start()

    print("=" * 50)
    print("PA_grinding 오케스트레이터 (ROS2 + TCP)")
    print("=" * 50)
    print(f"로봇: {HOST}")
    print()

    sockets = connect_sockets()
    print()

    if not sockets['Command']:
        print("[오류] Command 소켓 연결 필수. 종료합니다.")
        return

    try:
        while True:
            wp_status = f"{len(ros_node.waypoints)}개" if ros_node.path_received else "-"
            print()
            print("=" * 40)
            print(f"  로봇 상태: {last_status} ({status_text()})")
            print(f"  경로: {wp_status}")
            print("-" * 40)
            print("  1 = 종료")
            print("  2 = 스캔 위치 이동")
            print("  3 = (현재 조인트 수신)경로 전송 + 가공 준비")
            print("  4 = 가공 시작")
            print("  a = 자동 그라인딩")
            print("  q = 종료")
            print("=" * 40)

            cmd = input("입력: ").strip()

            if cmd == 'q':
                break
            elif cmd == 'a':
                print("\n[AUTO] 자동 그라인딩 시작 (Ctrl+C로 중단)")
                print("  카메라가 실행 중이어야 합니다 (real)")
                auto_grind_loop(sockets, ros_node)
            elif cmd in ['1', '2', '3', '4']:
                cmd_num = int(cmd)
                send_command(sockets['Command'], cmd_num)

                if cmd_num == 3:
                    time.sleep(0.1)
                    if ros_node.path_received and ros_node.waypoints:
                        print(f"[ROS2] 경로 전송 ({len(ros_node.waypoints)} waypoints)")
                        send_waypoints(sockets['Motion'], ros_node.waypoints)
                    else:
                        print("[경고] 경로 없음")

                wait_feedback(sockets['Feedback'], CMD_DONE_STATUS[cmd_num])
            else:
                print("잘못된 입력")

    except KeyboardInterrupt:
        print("\n[중단] Ctrl+C")
    except Exception as e:
        print(f"[오류] {e}")
    finally:
        close_sockets(sockets)
        ros_node.destroy_node()
        rclpy.shutdown()
        print("[완료]")


if __name__ == "__main__":
    main()
