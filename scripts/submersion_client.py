import socket
import pickle
import struct
import time
import numpy as np
import sys
import cv2
import lunar_tools as lt
from rtd.utils.prompt_provider import PromptProviderMicrophone, PromptProviderTxtFile
from rtd.utils.audio_detector import AudioDetector
from rtd.utils.oscillators import Oscillator
from rtd.utils.image_utils import gen_random_image
import threading
from dotenv import load_dotenv
import os
from rtd.utils.fft_analyzer import get_stream_analyzer  # Add import for FFT analyzer
from collections import deque

load_dotenv(override=True)

from rtd.utils.input_image import InputImageProcessor, AcidProcessor
from rtd.utils.compression_helpers import send_compressed, recv_compressed
# from rtd.utils.optical_flow import OpticalFlowEstimator
# from rtd.utils.posteffect import Posteffect

import random

from diffusers.utils import load_image

###############################################################################
# SubmersionClient
###############################################################################
class SubmersionClient:
    def __init__(self, server_host="localhost", server_port=8189):
        self.server_host = server_host
        self.server_port = server_port

        # Camera settings.
        self.shape_hw_cam = (1024, 1024)
        # Renderer settings.
        self.width_render = 1024
        self.height_render = 1024
        self.do_fullscreen = True

        # Initialize LT camera, meta input, and renderer.
        self.cam = lt.WebCam(shape_hw=self.shape_hw_cam)
        self.meta_input = lt.MetaInput()
        self.renderer = lt.Renderer(
            width=self.width_render,
            height=self.height_render,
            backend="opencv",
            do_fullscreen=self.do_fullscreen,
        )

        # Initialize the microphone prompt and speech detection.
        self.speech_detector = lt.Speech2Text()
        self.prompt_provider_microphone = PromptProviderMicrophone()
        self.prompt_provider_txt_file = PromptProviderTxtFile(os.getcwd()+"/materials/prompts/dancing_fibers.txt")
        self.audio_detector = AudioDetector()
        self.oscillator = Oscillator()
        self.fps_tracker = lt.FPSTracker()

        # Initialize the FFT audio analyzer
        self.fft_analyzer = get_stream_analyzer()

        # Initialize local processors for optical flow and posteffect processing.
        # self.opt_flow_estimator = OpticalFlowEstimator(use_ema=False)
        # self.posteffect_processor = Posteffect()
        self.input_image_processor = InputImageProcessor(do_human_seg=False)  # For computing human segmentation locally.

        # Shared variables for asynchronous networking.
        self.network_lock = threading.Lock()
        self.latest_cam_image = None
        self.latest_remote_diffusion = None
        
        # Initialize counter variables for incrementing values
        self.zoom_factor_value = 1.0  # Starting at 1.0
        self.x_shift_value = 0.0     # Starting at 0.0
        self.y_shift_value = 0.0     # Starting at 0.0
        
        # Variables for frequency spectrum analysis
        self.low_bin_baseline = deque(maxlen=30)  # stores last 30 values for rolling baseline
        self.high_bin_baseline = deque(maxlen=30)  # stores last 30 values for rolling baseline
        self.low_bin_sensitivity = 0.1  # How much low frequencies affect zoom out
        self.high_bin_sensitivity = 0.1  # How much high frequencies affect zoom in

        # Connect to the server.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((self.server_host, self.server_port))
        print(f"Connected to server at {self.server_host}:{self.server_port}")

    def send_msg(self, sock, msg):
        """Send a length-prefixed message."""
        msg = struct.pack("!I", len(msg)) + msg
        sock.sendall(msg)

    def recvall(self, sock, n):
        """Helper: receive exactly n bytes."""
        data = b""
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def recv_msg(self, sock):
        """Receive a length-prefixed message."""
        raw_msglen = self.recvall(sock, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack("!I", raw_msglen)[0]
        return self.recvall(sock, msglen)
        
    def process_frequency_bins(self, binned_fft):
        """
        Process frequency bins to adjust zoom factor based on changes in low and high frequency bands.
        With a balancing mechanism to return to neutral zoom (1.0) when there is no significant activity.
        
        Args:
            binned_fft: List containing the frequency bins (low, mid, high)
            
        Returns:
            float: Adjusted zoom factor value
        """
        if not binned_fft or len(binned_fft) < 3:
            return self.zoom_factor_value
            
        low_bin = binned_fft[0]
        high_bin = binned_fft[2]
        
        # Update the rolling baselines
        self.low_bin_baseline.append(low_bin)
        self.high_bin_baseline.append(high_bin)
        
        # Calculate baseline averages (if we have enough data)
        if len(self.low_bin_baseline) > 5 and len(self.high_bin_baseline) > 5:
            low_baseline_avg = sum(self.low_bin_baseline) / len(self.low_bin_baseline)
            high_baseline_avg = sum(self.high_bin_baseline) / len(self.high_bin_baseline)
            
            # Calculate delta from baseline as percentage changes
            if low_baseline_avg > 0:
                low_delta_pct = max(0, (low_bin - low_baseline_avg) / low_baseline_avg)
            else:
                low_delta_pct = 0
                
            if high_baseline_avg > 0:
                high_delta_pct = max(0, (high_bin - high_baseline_avg) / high_baseline_avg)
            else:
                high_delta_pct = 0
            
            # Normalize deltas to a reasonable range for zoom factor adjustments (0.01-0.05 per frame)
            # Cap percentage changes to avoid extreme reactions
            low_delta_pct = min(low_delta_pct, 1.0)  # Cap at 100% increase
            high_delta_pct = min(high_delta_pct, 1.0)  # Cap at 100% increase
            
            # Scale percentage changes to small increments appropriate for zoom
            zoom_out_factor = low_delta_pct * self.low_bin_sensitivity
            zoom_in_factor = high_delta_pct * self.high_bin_sensitivity
            
            # Apply adjustments to zoom factor
            # High frequencies increase zoom (zoom in)
            # Low frequencies decrease zoom (zoom out)
            zoom_adjustment = zoom_in_factor - zoom_out_factor
            
            # Calculate activity level to determine if we should return to neutral
            total_activity = low_delta_pct + high_delta_pct
            
            # If there's significant activity, apply the calculated adjustment
            # Otherwise gradually return to neutral (1.0)
            if max(low_delta_pct, high_delta_pct) > 0.9:  # Threshold for considering activity significant
                # Apply normal adjustment based on frequency analysis
                new_zoom = self.zoom_factor_value + zoom_adjustment
            else:
                # Return to neutral (1.0) gradually when no significant activity
                # Apply a small correction toward 1.0 (neutral position)
                rebalance_rate = 0.005  # Small step toward neutral per frame
                if self.zoom_factor_value > 1.0:
                    new_zoom = self.zoom_factor_value - rebalance_rate
                elif self.zoom_factor_value < 1.0:
                    new_zoom = self.zoom_factor_value + rebalance_rate
                else:
                    new_zoom = 1.0
                    
                # Print occasional debug info about rebalancing
                print(f"No significant audio activity. Rebalancing zoom: {self.zoom_factor_value:.2f} -> {new_zoom:.2f}")
            
            # Keep within reasonable bounds (0.8 to 1.5)
            new_zoom = max(0.8, min(1.5, new_zoom))
            
            # Print some debug info occasionally
            print(f"Frequency bins - Low: {low_bin:.2f} (Δ%: {low_delta_pct:.2f}), High: {high_bin:.2f} (Δ%: {high_delta_pct:.2f})")
            print(f"Zoom adjustment: {zoom_adjustment:.4f}, New zoom: {new_zoom:.2f}")
            
            return new_zoom
            
        return self.zoom_factor_value  # Return current value if not enough baseline data

    def network_loop(self):
        """Asynchronous thread for sending camera images to the server and receiving img_diffusion."""
        while True:
            try:
                with self.network_lock:
                    cam_img = self.latest_cam_image.copy() if self.latest_cam_image is not None else None
                
                # Get FFT audio features
                raw_fftx, raw_fft, binned_fftx, binned_fft = self.fft_analyzer.get_audio_features()
                
                # Process frequency bins to adjust zoom factor based on audio
                if binned_fft is not None and hasattr(binned_fft, 'tolist'):
                    self.zoom_factor_value = self.process_frequency_bins(binned_fft.tolist())
                
                # # Update incrementing values for x_shift and y_shift
                # self.x_shift_value += 0.1
                # if self.x_shift_value > 5.0:  # Reset if exceeds upper limit
                #     self.x_shift_value = -1.0
                
                # self.y_shift_value += 0.1
                # if self.y_shift_value > 5.0:  # Reset if exceeds upper limit
                #     self.y_shift_value = -1.0

                payload = {
                    "do_human_seg": self.meta_input.get(akai_lpd8="B1", akai_midimix="E3", button_mode="toggle", val_default=False),
                    "acid_strength": self.meta_input.get(akai_lpd8="E0", akai_midimix="C0", val_min=0, val_max=1.0, val_default=0.45),
                    "acid_strength_foreground": self.meta_input.get(akai_lpd8="E1", akai_midimix="C1", val_min=0, val_max=1.0, val_default=0.4),
                    "coef_noise": self.meta_input.get(akai_lpd8="F0", akai_midimix="C2", val_min=0, val_max=0.3, val_default=0.05),
                    "zoom_factor": self.meta_input.get(akai_lpd8="F1", akai_midimix="H2", val_min=0.5, val_max=1.5, val_default=self.zoom_factor_value),
                    "x_shift": int(self.meta_input.get(akai_midimix="H0", val_min=-50, val_max=50, val_default=self.x_shift_value)),
                    "y_shift": int(self.meta_input.get(akai_midimix="H1", val_min=-50, val_max=50, val_default=self.y_shift_value)),
                    "color_matching": self.meta_input.get(akai_lpd8="G0", akai_midimix="G0", val_min=0, val_max=1, val_default=0.5),
                    "mic_prompt": self.prompt_provider_microphone.get_current_prompt() if self.prompt_provider_microphone.handle_unmute_button(self.meta_input.get(akai_lpd8="A1", akai_midimix="A3", button_mode="held_down")) else None,
                    "txt_file_prompt": "a blue dog and a red monkey", #self.prompt_provider_txt_file.get_current_prompt() if self.meta_input.get(akai_lpd8="C0", akai_midimix="A4", button_mode="pressed_once") else None,
                    "dynamic_transcript": self.speech_detector.transcript if self.speech_detector.handle_unmute_button(self.meta_input.get(akai_lpd8="A0", akai_midimix="B3", button_mode="held_down")) else None,
                    "brightness": self.meta_input.get(akai_midimix="A2", val_min=0.0, val_max=2, val_default=1.0),
                    "do_infrared_colorize": self.meta_input.get(akai_lpd8="D0", akai_midimix="H4", button_mode="toggle", val_default=False),
                    "dyn_prompt_restore_backup": self.meta_input.get(akai_midimix="F3", button_mode="released_once"),
                    "dyn_prompt_del_current": self.meta_input.get(akai_midimix="F4", button_mode="released_once"),
                    "prompt_transition_time": self.meta_input.get(akai_lpd8="G1", val_min=1, val_max=20, val_default=1.0),
                    # Add FFT audio features to the payload
                    # "raw_fftx": raw_fftx.tolist() if hasattr(raw_fftx, 'tolist') else raw_fftx,
                    # "raw_fft": raw_fft.tolist() if hasattr(raw_fft, 'tolist') else raw_fft,
                    # "binned_fftx": binned_fftx.tolist() if hasattr(binned_fftx, 'tolist') else binned_fftx,
                    "binned_fft": binned_fft.tolist() if hasattr(binned_fft, 'tolist') else binned_fft,
                }
                data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
                self.send_msg(self.sock, data)

                # print(f"Sending payload: {payload}")

                if cam_img is not None:
                    if cam_img.shape[:2] != self.cam.shape_hw:
                        desired_width = self.cam.shape_hw[1]
                        desired_height = self.cam.shape_hw[0]
                        cam_img = cv2.resize(cam_img, (desired_width, desired_height))
                    print("Sending compressed image to server...")
                    send_compressed(self.sock, cam_img, quality=90)
                    print("Image sent successfully")
                else:
                    print("Warning: No camera image available to send")
                    # Send a dummy image to keep the protocol in sync
                    dummy_img = np.zeros((self.cam.shape_hw[0], self.cam.shape_hw[1], 3), dtype=np.uint8)
                    send_compressed(self.sock, dummy_img, quality=90)

                print("Sent payload")

                print("Waiting for processed image from server...")
                processed_image = recv_compressed(self.sock)
                print("Received processed image", processed_image.shape)
                
                if processed_image is not None:
                    with self.network_lock:
                        self.latest_remote_diffusion = processed_image
                else:
                    print("Disconnected from server in network thread")
                    break
            except Exception as e:
                print("Error in network thread:", e)
                import traceback
                traceback.print_exc()
                break
            time.sleep(0.01)  # Small sleep to prevent a tight loop

    def run(self):
        # Start the network communication loop in a separate thread.
        network_thread = threading.Thread(target=self.network_loop, daemon=True)
        network_thread.start()
        while True:
            self.fps_tracker.start_segment("Client Processing")
            t_processing_start = time.time()

            self.fps_tracker.start_segment("Camera Capture")
            img_cam = self.cam.get_img()

            with self.network_lock:
                self.latest_cam_image = img_cam.copy()

            # Perform local processing: compute human segmentation and optical flow.
            self.fps_tracker.start_segment("Input Image Processing")
            img_proc, human_seg_mask = self.input_image_processor.process(img_cam.copy())
            if human_seg_mask is None or not np.any(human_seg_mask):
                human_seg_mask = np.ones_like(img_proc, dtype=np.float32) / 255

            self.fps_tracker.start_segment("Optical Flow")
            opt_flow = None
            # opt_flow = self.opt_flow_estimator.get_optflow(img_cam.copy(), low_pass_kernel_size=55, window_length=55)

            with self.network_lock:
                remote_diff = self.latest_remote_diffusion.copy() if self.latest_remote_diffusion is not None else None

            # Retrieve postprocessing parameters.
            postproc_func_coef1 = self.meta_input.get(akai_lpd8="H0", akai_midimix="G1", val_min=0, val_max=1, val_default=0.5)
            postproc_func_coef2 = self.meta_input.get(akai_lpd8="H1", akai_midimix="G2", val_min=0, val_max=1, val_default=0.5)
            postproc_mod_button1 = self.meta_input.get(akai_midimix="G4", button_mode="toggle", val_default=True)
            sound_volume = 0
            if self.meta_input.get(akai_midimix="D4", button_mode="toggle", val_default=False):
                sound_volume = self.audio_detector.get_last_volume()

            # If no remote diffusion image is available, fall back to the camera image.
            if remote_diff is None:
                remote_diff = img_cam.copy()

            if opt_flow is None:
                opt_flow = np.zeros(remote_diff.shape, dtype=remote_diff.dtype)
                opt_flow = opt_flow[:,:,::2]

            self.fps_tracker.start_segment("Post Processing")
            # output_to_render, _ = self.posteffect_processor.process(
            #     remote_diff,
            #     human_seg_mask.astype(np.float32) / 255,
            #     opt_flow,
            #     postproc_func_coef1,
            #     postproc_func_coef2,
            #     postproc_mod_button1,
            #     sound_volume,
            # )

            self.fps_tracker.start_segment("Rendering")
            self.renderer.render(remote_diff)

            t_processing = time.time() - t_processing_start
            self.fps_tracker.print_fps()

        self.sock.close()

###############################################################################
# Main entry point
###############################################################################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} [server|client]")
        sys.exit(1)

    role = sys.argv[1].lower()
    if len(sys.argv) > 2:
        server_ip = sys.argv[2].lower()
    else:
        server_ip = "localhost"

    if role == "client":
        client = SubmersionClient(server_host=server_ip)
        client.run()
    else:
        print("Invalid mode. Use 'server' or 'client'.")
        print("Invalid mode. Use 'server' or 'client'.")
        print("Invalid mode. Use 'server' or 'client'.")
