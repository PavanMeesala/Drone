
from kivy.app import App
from kivy.graphics import Color, Rectangle
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window
from kivy.utils import platform
from kivy.uix.label import Label
from kivy.uix.image import Image

import socket
import cv2
import numpy as np
import json

import os.path
from kivy.resources import resource_add_path
KV_PATH = os.path.realpath(os.path.dirname(__file__))
resource_add_path(KV_PATH)

from kivy.lang import Builder
Builder.load_file("joystickpad.kv")
Builder.load_file("joystick.kv")

from kivy.uix.widget import Widget
from kivy.properties import(BooleanProperty, NumericProperty,
                            ListProperty, ReferenceListProperty)
import math

OUTLINE_ZERO = 0.00000000001
# replaces user's 0 value for outlines, avoids invalid width exception


class Joystick(Widget):
    '''The joystick base is comprised of an outer circle & an inner circle.
       The joystick pad is another circle,
           which the user can move within the base.
       All 3 of these elements can be styled independently
           to create different effects.
       All coordinate properties are based on the
           position of the joystick pad.'''

    '''####################################################################'''
    '''#####   >   Properties (Customizable)   ############################'''
    '''####################################################################'''

    outer_size = NumericProperty(1)
    inner_size = NumericProperty(0.75)
    pad_size = NumericProperty(0.5)
    '''Sizes are defined by percentage,
           1.0 being 100%, of the total widget size.
        The smallest value of widget.width & widget.height
           is used as a baseline for these percentages.'''

    outer_background_color = ListProperty([0.75, 0.75, 0.75, 1])
    inner_background_color = ListProperty([0.75, 0.75, 0.75, 1])
    pad_background_color = ListProperty([0.4, 0.4, 0.4, 1])
    '''Background colors for the joystick base & pad'''

    outer_line_color = ListProperty([0.25, 0.25, 0.25, 1])
    inner_line_color = ListProperty([0.7, 0.7, 0.7, 1])
    pad_line_color = ListProperty([0.35, 0.35, 0.35, 1])
    '''Border colors for the joystick base & pad'''

    outer_line_width = NumericProperty(0.01)
    inner_line_width = NumericProperty(0.01)
    pad_line_width = NumericProperty(0.01)
    '''Outline widths for the joystick base & pad.
       Outline widths are defined by percentage,
           1.0 being 100%, of the total widget size.'''

    sticky = BooleanProperty(False)
    '''When False, the joystick will snap back to center on_touch_up.
       When True, the joystick will maintain its final position
           at the time of on_touch_up.'''

    '''####################################################################'''
    '''#####   >   Properties (Read-Only)   ###############################'''
    '''####################################################################'''

    pad_x = NumericProperty(0.0)
    pad_y = NumericProperty(0.0)
    pad = ReferenceListProperty(pad_x, pad_y)
    '''pad values are touch coordinates in relation to
           the center of the joystick.
       pad_x & pad_y return values between -1.0 & 1.0.
       pad returns a tuple of pad_x & pad_y, and is the best property to
           bind to in order to receive updates from the joystick.'''

    @property
    def magnitude(self):
        return self._magnitude
    '''distance of the pad, between 0.0 & 1.0,
           from the center of the joystick.'''

    @property
    def radians(self):
        return self._radians
    '''degrees of the pad, between 0.0 & 360.0, in relation to the x-axis.'''

    @property
    def angle(self):
        return math.degrees(self.radians)
    '''position of the pad in radians, between 0.0 & 6.283,
           in relation to the x-axis.'''

    '''magnitude, radians, & angle can be used to
           calculate polar coordinates'''

    '''####################################################################'''
    '''#####   >   Properties (Private)   #################################'''
    '''####################################################################'''

    _outer_line_width = NumericProperty(OUTLINE_ZERO)
    _inner_line_width = NumericProperty(OUTLINE_ZERO)
    _pad_line_width = NumericProperty(OUTLINE_ZERO)

    _total_diameter = NumericProperty(0)
    _total_radius = NumericProperty(0)

    _inner_diameter = NumericProperty(0)
    _inner_radius = NumericProperty(0)

    _outer_diameter = NumericProperty(0)
    _outer_radius = NumericProperty(0)

    _magnitude = 0

    @property
    def _radians(self):
        if not(self.pad_y and self.pad_x):
            return 0
        arc_tangent = math.atan(self.pad_y / self.pad_x)
        if self.pad_x > 0 and self.pad_y > 0:    # 1st Quadrant
            return arc_tangent
        elif self.pad_x > 0 and self.pad_y < 0:  # 4th Quadrant
            return (math.pi * 2) + arc_tangent
        else:                                    # 2nd & 3rd Quadrants
            return math.pi + arc_tangent

    @property
    def _radius_difference(self):
        return (self._total_radius - self.ids.pad._radius)

    '''####################################################################'''
    '''#####   >   Pad Control   ##########################################'''
    '''####################################################################'''

    def move_pad(self, touch, from_touch_down):
        td = TouchData(self, touch)
        if td.is_external and from_touch_down:
            touch.ud['joystick'] = None
            return False
        elif td.in_range:
            self._update_coordinates_from_internal_touch(touch, td)
            return True
        elif not(td.in_range):
            self._update_coordinates_from_external_touch(td)
            return True

    def center_pad(self):
        self.ids.pad.center = self.center
        self._magnitude = 0
        self.pad_x = 0
        self.pad_y = 0

    def _update_coordinates_from_external_touch(self, touchdata):
        td = touchdata
        pad_distance = self._radius_difference * (1.0 / td.relative_distance)
        x_distance_offset = -td.x_distance * pad_distance
        y_distance_offset = -td.y_distance * pad_distance
        x = self.center_x + x_distance_offset
        y = self.center_y + y_distance_offset
        radius_offset = pad_distance / self._radius_difference
        self.pad_x = td.x_offset * radius_offset
        self.pad_y = td.y_offset * radius_offset
        self._magnitude = 1.0
        self.ids.pad.center = (x, y)

    def _update_coordinates_from_internal_touch(self, touch, touchdata):
        td = touchdata
        self.pad_x = td.x_offset / self._radius_difference
        self.pad_y = td.y_offset / self._radius_difference
        self._magnitude = td.relative_distance / \
            (self._total_radius - self.ids.pad._radius)
        self.ids.pad.center = (touch.x, touch.y)

    '''####################################################################'''
    '''#####   >   Layout Events   ########################################'''
    '''####################################################################'''

    def do_layout(self):
        if 'pad' in self.ids:
            size = min(*self.size)
            self._update_outlines(size)
            self._update_circles(size)
            self._update_pad()

    def on_size(self, *args):
        self.do_layout()

    def on_pos(self, *args):
        self.do_layout()

    def add_widget(self, widget):
        super(Joystick, self).add_widget(widget)
        self.do_layout()

    def remove_widget(self, widget):
        super(Joystick, self).remove_widget(widget)
        self.do_layout()

    def _update_outlines(self, size):
        self._outer_line_width = (self.outer_line_width * size) \
            if(self.outer_line_width) else(OUTLINE_ZERO)
        self._inner_line_width = (self.inner_line_width * size) \
            if(self.inner_line_width) else(OUTLINE_ZERO)
        self.ids.pad._line_width = (self.pad_line_width * size) \
            if(self.pad_line_width) else(OUTLINE_ZERO)

    def _update_circles(self, size):
        self._total_diameter = size
        self._total_radius = self._total_diameter / 2
        self._outer_diameter = \
            (self._total_diameter - self._outer_line_width) * self.outer_size
        self._outer_radius = self._outer_diameter / 2
        self.ids.pad._diameter = self._total_diameter * self.pad_size
        self.ids.pad._radius = self.ids.pad._diameter / 2
        self._inner_diameter = \
            (self._total_diameter - self._inner_line_width) * self.inner_size
        self._inner_radius = self._inner_diameter / 2

    def _update_pad(self):
        self.ids.pad.center = self.center
        self.ids.pad._background_color = self.pad_background_color
        self.ids.pad._line_color = self.pad_line_color

    '''####################################################################'''
    '''#####   >   Touch Events   #########################################'''
    '''####################################################################'''

    def on_touch_down(self, touch):
        if self.collide_point(touch.x, touch.y):
            touch.ud['joystick'] = self
            return self.move_pad(touch, from_touch_down=True)
        return super(Joystick, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if self._touch_is_active(touch):
            return self.move_pad(touch, from_touch_down=False)
        return super(Joystick, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        if self._touch_is_active(touch) and not(self.sticky):
            self.center_pad()
            return True
        return super(Joystick, self).on_touch_up(touch)

    def _touch_is_active(self, touch):
        return 'joystick' in touch.ud and touch.ud['joystick'] == self
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ListProperty


class JoystickPad(Widget):
    _diameter = NumericProperty(1)
    _radius = NumericProperty(1)
    _background_color = ListProperty([0, 0, 0, 1])
    _line_color = ListProperty([1, 1, 1, 1])
    _line_width = NumericProperty(1)

class TouchData:
    x_distance = None
    y_distance = None
    x_offset = None
    y_offset = None
    relative_distance = None
    is_external = None
    in_range = None

    def __init__(self, joystick, touch):
        self.joystick = joystick
        self.touch = touch
        self._calculate()

    def _calculate(self):
        js = self.joystick
        touch = self.touch
        x_distance = js.center_x - touch.x
        y_distance = js.center_y - touch.y
        x_offset = touch.x - js.center_x
        y_offset = touch.y - js.center_y
        relative_distance = ((x_distance ** 2) + (y_distance ** 2)) ** 0.5
        is_external = relative_distance > js._total_radius
        in_range = relative_distance <= js._radius_difference
        self._update(x_distance, y_distance, x_offset, y_offset,
                     relative_distance, is_external, in_range)

    def _update(self, x_distance, y_distance, x_offset, y_offset,
                relative_distance, is_external, in_range):
        self.x_distance = x_distance
        self.y_distance = y_distance
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.relative_distance = relative_distance
        self.is_external = is_external
        self.in_range = in_range




# Request permissions on Android
Config.set('graphics', 'orientation', 'landscape')
Window.clearcolor = (0, 0, 0, 0)  # Transparent window


class VideoBackground(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stream_buffer = b''
        self.sock = None
        self.connected = False
        self.object_detection = False
        self.detection_model = None
        self.class_names = []

        # Create the Image widget
        self.video_image = Image(
            size_hint=(1, 1),
            allow_stretch=True,
            keep_ratio=False,
            opacity=1
        )
        self.add_widget(self.video_image)

        Clock.schedule_interval(self.update_frame, 1.0 / 30)

        # Initialize object detection (using OpenCV's built-in Haar cascades as example)
        self.init_object_detection()

    def init_object_detection(self):
        """Initialize object detection models"""
        # Load a simple face detection model (replace with your preferred model)
        model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(model_path):
            self.detection_model = cv2.CascadeClassifier(model_path)
            self.class_names = ['face']  # Simple class names for this example
        else:
            print("Warning: Could not load detection model")

    def connect(self, host, port=8888):
        try:
            if self.sock:
                self.sock.close()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((host, port))
            self.connected = True
            print("Video connected successfully")
        except Exception as e:
            print(f"Video connection failed: {e}")
            self.connected = False

    def toggle_object_detection(self):
        """Toggle object detection on/off"""
        self.object_detection = not self.object_detection
        return self.object_detection

    def detect_objects(self, frame):
        """Perform object detection on the frame"""
        if self.detection_model is None:
            return frame

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect objects (using face detection as example)
        detections = self.detection_model.detectMultiScale(gray, 1.1, 4)

        # Draw bounding boxes
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            label = f"{self.class_names[0]}" if self.class_names else "object"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return frame

    def update_frame(self, dt):
        if not self.connected:
            return

        try:
            data = self.sock.recv(65536)
            if not data:
                self.connected = False
                return

            self.stream_buffer += data
            a = self.stream_buffer.find(b'\xff\xd8')
            b = self.stream_buffer.find(b'\xff\xd9')

            if a != -1 and b != -1:
                jpg = self.stream_buffer[a:b + 2]
                self.stream_buffer = self.stream_buffer[b + 2:]

                img_np = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                if frame is not None:
                    # Apply object detection if enabled
                    if self.object_detection and self.detection_model:
                        frame = self.detect_objects(frame)

                    buf = cv2.flip(frame, 0).tobytes()
                    texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                    texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                    self.video_image.texture = texture
        except Exception as e:
            print(f"Video error: {e}")
            self.connected = False


class ControlOverlay(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Make the overlay fully transparent
        with self.canvas.before:
            Color(0, 0, 0, 0)
            self.bg = Rectangle(pos=self.pos, size=self.size)

        self.bind(pos=self.update_bg, size=self.update_bg)
        self.setup_controls()

    def update_bg(self, *args):
        self.bg.pos = self.pos
        self.bg.size = self.size

    def setup_controls(self):
        # Left Joystick (Movement)
        self.left_joystick = Joystick(size_hint=(0.35, 0.35), pos_hint={'x': 0.05, 'y': 0.3})
        self.left_joystick.background_color = (1, 1, 1, 0.3)
        self.left_joystick.outline_color = (1, 1, 1, 0.5)
        self.add_widget(self.left_joystick)

        # Right Joystick (Semi-Transparent)
        self.right_joystick = Joystick(size_hint=(0.35, 0.35), pos_hint={'x': 0.65, 'y': 0.3})
        self.right_joystick.background_color = (1, 1, 1, 0.3)
        self.right_joystick.outline_color = (1, 1, 1, 0.5)
        self.add_widget(self.right_joystick)

        # Camera Joystick (Semi-Transparent)
        self.camera_joystick = Joystick(size_hint=(0.3, 0.3), pos_hint={'x': 0.75, 'y': 0.7})
        self.camera_joystick.background_color = (1, 1, 1, 0.3)
        self.camera_joystick.outline_color = (1, 1, 1, 0.5)
        self.add_widget(self.camera_joystick)

        # Start/Stop Button
        self.start_btn = Button(
            text="START DRONE",
            size_hint=(0.2, 0.1),
            pos_hint={'x': 0.3, 'y': 0.8},
            background_normal='',
            background_color=(0.2, 0.7, 0.2, 0.7),
            color=(1, 1, 1, 1),
            font_size='18sp',
            bold=True
        )
        self.add_widget(self.start_btn)

        # Object Detection Toggle Button
        self.detection_btn = Button(
            text="Enable Detection",
            size_hint=(0.2, 0.1),
            pos_hint={'x': 0.5, 'y': 0.8},
            background_normal='',
            background_color=(0.4, 0.4, 0.8, 0.7),
            color=(1, 1, 1, 1),
            font_size='18sp',
            bold=True
        )
        self.add_widget(self.detection_btn)

        # Menu Button
        self.menu_btn = Button(
            text="Menu",  # Gear emoji
            size_hint=(0.1, 0.1),
            pos_hint={'x': 0, 'y': 0.88},
            background_normal='',
            background_color=(0.5, 0.5, 0.5, 0.5),
            color=(1, 1, 1, 1),
            font_size='24sp'
        )
        self.add_widget(self.menu_btn)

        # Connection Status Label
        self.status_label = Label(
            text="Disconnected",
            size_hint=(0.3, 0.05),
            pos_hint={'x': 0.35, 'y': 0.05},
            color=(1, 0, 0, 1),
            bold=True
        )
        self.add_widget(self.status_label)


class DroneControllerApp(App):
    def build(self):
        self.root = FloatLayout()

        # 1. Video Background (bottom layer)
        self.video_bg = VideoBackground()
        self.root.add_widget(self.video_bg)

        # 2. Control Overlay (top layer with all transparent controls)
        self.controls = ControlOverlay()
        self.root.add_widget(self.controls)

        # Setup button bindings
        self.setup_bindings()

        return self.root

    def setup_bindings(self):
        # Bind menu button to open connection dialog
        self.controls.menu_btn.bind(on_release=self.show_connection_dialog)

        # Bind start button to toggle drone state
        self.controls.start_btn.bind(on_press=self.toggle_drone)

        # Bind detection button to toggle object detection
        self.controls.detection_btn.bind(on_press=self.toggle_detection)

        # Bind joysticks to send commands
        self.controls.left_joystick.bind(pad=self.send_joystick_data)
        self.controls.right_joystick.bind(pad=self.send_joystick_data)
        self.controls.camera_joystick.bind(pad=self.send_joystick_data)

    def toggle_detection(self, instance):
        detection_on = self.video_bg.toggle_object_detection()
        if detection_on:
            self.controls.detection_btn.text = "Disable Detection"
            self.controls.detection_btn.background_color = (0.8, 0.2, 0.2, 0.8)
        else:
            self.controls.detection_btn.text = "Enable Detection"
            self.controls.detection_btn.background_color = (0.4, 0.4, 0.8, 0.7)

    def show_connection_dialog(self, instance):
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)

        self.ip_input = TextInput(
            hint_text='Enter Raspberry Pi IP',
            multiline=False,
            size_hint=(1, 0.4),
            font_size='18sp'
        )
        content.add_widget(self.ip_input)

        connect_btn = Button(
            text="CONNECT",
            size_hint=(1, 0.4),
            background_normal='',
            background_color=(0.2, 0.5, 0.8, 1)
        )
        connect_btn.bind(on_release=self.connect_to_pi)
        content.add_widget(connect_btn)

        self.connection_popup = Popup(
            title="Drone Connection",
            content=content,
            size_hint=(0.8, 0.4)
        )
        self.connection_popup.open()

    def connect_to_pi(self, instance):
        pi_ip = self.ip_input.text.strip()
        if pi_ip:
            # Connect video stream
            self.video_bg.connect(pi_ip)

            # Connect control socket
            try:
                self.control_socket = socket.socket()
                self.control_socket.connect((pi_ip, 5000))
                self.controls.status_label.text = "Connected"
                self.controls.status_label.color = (0, 1, 0, 1)
                self.connection_popup.dismiss()
            except Exception as e:
                self.controls.status_label.text = f"Error: {str(e)}"
                self.controls.status_label.color = (1, 0, 0, 1)

    def toggle_drone(self, instance):
        if self.controls.start_btn.text == "START DRONE":
            self.controls.start_btn.text = "STOP DRONE"
            self.controls.start_btn.background_color = (0.8, 0.2, 0.2, 0.8)
            self.send_command("start")
        else:
            self.controls.start_btn.text = "START DRONE"
            self.controls.start_btn.background_color = (0.2, 0.7, 0.2, 0.7)
            self.send_command("stop")

    def send_command(self, command):
        if hasattr(self, 'control_socket'):
            try:
                self.control_socket.send((command + "\n").encode())
            except Exception as e:
                print(f"Command failed: {e}")

    def send_joystick_data(self, instance, value):
        if hasattr(self, 'control_socket'):
            data = {
                'left': self.controls.left_joystick.pad if self.controls.left_joystick.pad else [0, 0],
                'right': self.controls.right_joystick.pad if self.controls.right_joystick.pad else [0, 0],
                'camera': self.controls.camera_joystick.pad if self.controls.camera_joystick.pad else [0, 0]
            }
            try:
                if self.controls.start_btn.text == "STOP DRONE":
                    # Only send joystick data if the drone is started
                    message = json.dumps(data)
                    self.control_socket.send((str(data) + "\n").encode())
            except Exception as e:
                print(f"Joystick data failed: {e}")


if __name__ == '__main__':
    DroneControllerApp().run()