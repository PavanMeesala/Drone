import socket
import threading
import time
import smbus
import RPi.GPIO as GPIO
import subprocess
from math import fabs
import json

# === IMU Setup ===
bus = smbus.SMBus(1)
MPU_ADDR = 0x68
bus.write_byte_data(MPU_ADDR, 0x6B, 0)  # Wake up MPU

def read_word(reg):
    high = bus.read_byte_data(MPU_ADDR, reg)
    low = bus.read_byte_data(MPU_ADDR, reg + 1)
    val = (high << 8) + low
    return val - 65536 if val > 32767 else val

def get_imu_data():
    return (
        read_word(0x3B) / 16384.0,  # ax
        read_word(0x3D) / 16384.0,  # ay
        read_word(0x3F) / 16384.0,  # az
        read_word(0x43) / 131.0,    # gx
        read_word(0x45) / 131.0,    # gy
        read_word(0x47) / 131.0     # gz
    )

# === PID Controller ===
class PID:
    def __init__(self, kp, ki, kd, max_output=50):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max = max_output
        self.reset()

    def reset(self):
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return max(-self.max, min(self.max, output))

# === Motor Setup ===
motor_pins = [17, 18, 27, 22]
GPIO.setmode(GPIO.BCM)
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)
pwms = [GPIO.PWM(pin, 400) for pin in motor_pins]
for pwm in pwms:
    pwm.start(0)

def set_motor_speeds(speeds):
    for pwm, speed in zip(pwms, speeds):
        pwm.ChangeDutyCycle(max(0, min(100, speed)))

# === Video Streaming ===
def video_stream():
    HOST = '0.0.0.0'
    PORT = 8888

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"Video stream started on {HOST}:{PORT}")

        while True:
            try:
                conn, addr = s.accept()
                print(f"Video client connected from {addr}")
                with conn:
                    cmd = [
                        "libcamera-vid", "-t", "0",
                        "--width", "640", "--height", "480",
                        "--framerate", "30", "--codec", "mjpeg",
                        "--quality", "85", "--inline", "--flush",
                        "-n", "-o", "-"
                    ]

                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                    try:
                        while True:
                            data = process.stdout.read(65536)
                            if not data:
                                break
                            conn.sendall(data)
                    except (ConnectionResetError, BrokenPipeError):
                        print("Video client disconnected")
                    finally:
                        process.terminate()
            except Exception as e:
                print(f"Video stream error: {e}")
                time.sleep(1)

# === Flight Controller ===
class FlightController:
    def __init__(self):
        self.armed = False
        self.throttle = 0
        self.target_throttle = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.last_update = time.time()
        self.hover_throttle = 50

        self.pid_roll = PID(1.2, 0.05, 0.2)
        self.pid_pitch = PID(1.2, 0.05, 0.2)
        self.pid_yaw = PID(1.0, 0.01, 0.1)

    def update(self, dt):
        throttle_diff = self.target_throttle - self.throttle
        if fabs(throttle_diff) > 1.0:
            self.throttle += throttle_diff * 0.1

        if not self.armed:
            set_motor_speeds([0, 0, 0, 0])
            return

        try:
            ax, ay, az, gx, gy, gz = get_imu_data()
        except Exception as e:
            print(f"IMU read error: {e}")
            return

        if self.throttle > 5:
            roll_corr = self.pid_roll.compute(ax - self.roll, dt)
            pitch_corr = self.pid_pitch.compute(ay - self.pitch, dt)
            yaw_corr = self.pid_yaw.compute(gz - self.yaw, dt)
        else:
            roll_corr = pitch_corr = yaw_corr = 0

        m1 = self.throttle - roll_corr + pitch_corr + yaw_corr
        m2 = self.throttle + roll_corr + pitch_corr - yaw_corr
        m3 = self.throttle + roll_corr - pitch_corr + yaw_corr
        m4 = self.throttle - roll_corr - pitch_corr - yaw_corr
        #print(m1,m2,m3,m4)
        set_motor_speeds([m1, m2, m3, m4])

def handle_client(conn, addr, controller):
    print(f"Flight client connected from {addr}")
    try:
        with conn:
            while True:
                data = conn.recv(1024).decode().strip()
                #print(f"Raw data: {data}")

                if not data:
                    break

                controller.last_update = time.time()

                if data == "start":
                    if not controller.armed:
                        controller.armed = True
                        controller.target_throttle = controller.hover_throttle
                        print("Motors armed and hovering")
                elif data == "stop":
                    controller.armed = False
                    controller.target_throttle = 0
                    print("Motors disarmed")
                elif data.startswith("{"):
                    try:
                        js_data = json.loads(data)
                        throttle_input = js_data['left'][1]
                        if throttle_input > 0:
                            controller.target_throttle = controller.hover_throttle + throttle_input * 30
                        else:
                            controller.target_throttle = controller.hover_throttle + throttle_input * 20

                        controller.roll = js_data['right'][0] * 0.5
                        controller.pitch = js_data['right'][1] * 0.5
                        controller.yaw = js_data['left'][0] * 0.3
                    except json.JSONDecodeError:
                        print("Invalid JSON received")
    except Exception as e:
        print(f"Client error: {e}")
    finally:
        print("Client disconnected")

def flight_controller_server():
    HOST = '0.0.0.0'
    PORT = 5000

    controller = FlightController()

    video_thread = threading.Thread(target=video_stream, daemon=True)
    video_thread.start()

    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"Flight controller server started on {HOST}:{PORT}")

        def controller_loop():
            last_time = time.time()
            while True:
                dt = time.time() - last_time
                last_time = time.time()
                controller.update(dt)
                time.sleep(0.02)

        threading.Thread(target=controller_loop, daemon=True).start()

        try:
            while True:
                conn, addr = s.accept()
                threading.Thread(target=handle_client, args=(conn, addr, controller), daemon=True).start()
        except KeyboardInterrupt:
            print("\nShutting down...")
            for pwm in pwms:
                pwm.stop()
            GPIO.cleanup()

if __name__ == "__main__":
    flight_controller_server()

