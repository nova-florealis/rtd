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
import threading
from dotenv import load_dotenv
import os

load_dotenv(override=True)

if len(sys.argv) > 1 and sys.argv[1].lower() == "server":
    from rtd.sdxl_turbo.diffusion_engine import DiffusionEngine
    from rtd.sdxl_turbo.simple_diffusion_engine import SimpleDiffusionEngine
    from rtd.sdxl_turbo.embeddings_mixer import EmbeddingsMixer
    from rtd.dynamic_processor.processor_dynamic_module import DynamicProcessor
    from rtd.utils.input_image import InputImageProcessor, AcidProcessor
    # Even though these are imported for the server branch, note that the server will no longer
    # perform optical flow or postprocessing.
    from rtd.utils.optical_flow import OpticalFlowEstimator
    from rtd.utils.posteffect import Posteffect

from rtd.utils.input_image import InputImageProcessor, AcidProcessor
from rtd.utils.compression_helpers import send_compressed, recv_compressed
from rtd.utils.optical_flow import OpticalFlowEstimator
from rtd.utils.posteffect import Posteffect

###############################################################################
# SubmersionServer
###############################################################################
class SubmersionServer:
    def __init__(self, host="0.0.0.0", port=8189, device="cuda:0", do_diffusion=True, do_compile=True, bounce=False):
        self.host = host
        self.port = port
        self.device = device
        self.do_diffusion = do_diffusion
        self.do_compile = do_compile
        self.bounce = bounce  # If True, the server will simply echo back the received image without processing

        # These dimensions MUST match the submersion pipeline settings.
        self.height_diffusion = int((384 + 96) * 1.0)
        self.width_diffusion = int((512 + 128) * 1.0)

        # Create and bind the server socket.
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Disable Nagle's algorithm to reduce latency.
        self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"SubmersionServer listening on {self.host}:{self.port}")

        if not self.bounce:
            # Initialize image processing modules.
            self.input_image_processor = InputImageProcessor(device=device)
            self.input_image_processor.set_flip(do_flip=True, flip_axis=1)

            self.acid_processor = AcidProcessor(
                height_diffusion=self.height_diffusion,
                width_diffusion=self.width_diffusion,
                device=device,
            )
            # Dynamic processing is removed since postprocessing is now handled client‐side.
            # self.dynamic_processor = DynamicProcessor()

            # Removed optical flow and posteffect initializations from the server.
            # self.opt_flow_estimator = OpticalFlowEstimator(use_ema=False)
            # self.posteffect_processor = Posteffect()

            self.de_img = SimpleDiffusionEngine(
                # hf_model="sd-community/sdxl-flash",
                use_image2image=True,
                height_diffusion_desired=self.height_diffusion,
                width_diffusion_desired=self.width_diffusion,
                do_compile=self.do_compile,
                do_diffusion=self.do_diffusion,
                device=device,
                use_lightning=True,
            )

            self.de_img.set_guidance_scale(0.0) #Flash - 1.2
            self.de_img.set_strength(0.45) # Flash 0.4 // Default - 1 / self.de_img.num_inference_steps + 0.00001)
            self.de_img.set_num_inference_steps(4) # Flash - 10

            if self.do_diffusion:
                self.em = EmbeddingsMixer(self.de_img.pipe)
                init_prompt = 'A rainbow spectrum entity human figure in a dark void.'
                self.embeds = self.em.encode_prompt(init_prompt)
                self.embeds_source = self.em.clone_embeddings(self.embeds)
                self.embeds_target = self.em.clone_embeddings(self.embeds)
                self.de_img.set_embeddings(self.embeds)
                self.fract_blend_embeds = 1.0  # Start with fully blended embedding
                self.transition_start_time = None

            # Removed storage of last diffusion image as it is not needed here.
            # self.last_diffused = None

            self.fps_tracker = lt.FPSTracker()

            print("Submersion server ready.")
        else:
            print("Bounce mode enabled: Server will echo the received image without processing.")
            self.fps_tracker = lt.FPSTracker()

        # Create output directory
        os.makedirs("output", exist_ok=True)

    def recvall(self, sock, n):
        """Helper: receive exactly n bytes from the socket."""
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

    def send_msg(self, sock, msg):
        """Send a length-prefixed message."""
        msg = struct.pack("!I", len(msg)) + msg
        sock.sendall(msg)

    def permute_prompt(self, prompt):
        """Shuffles the words in a prompt while preserving basic structure."""
        if not prompt:
            return prompt
        words = prompt.split()
        # Don't shuffle if there's only one word
        if len(words) <= 1:
            return prompt
        # Shuffle the words
        np.random.shuffle(words)
        return ' '.join(words)

    def handle_client(self, client_sock, addr):
        print(f"Connected by {addr}")
        while True:
            try:
                self.fps_tracker.start_segment("Receive Data")
                data = self.recv_msg(client_sock)
                if data is None:
                    print("Client disconnected")
                    break

                # Unpickle the received payload.
                payload = pickle.loads(data)

                if not self.bounce:
                    self.fps_tracker.start_segment("Process Prompts")
                    mic_prompt = payload.get("mic_prompt")
                    if mic_prompt and self.do_diffusion:
                        print(f"New microphone prompt received: {mic_prompt}")
                        self.transition_start_time = time.time()
                        self.embeds_source = self.em.clone_embeddings(self.embeds)
                        self.embeds_target = self.em.encode_prompt(mic_prompt)
                        self.fract_blend_embeds = 0.0

                    txt_file_prompt = payload.get("txt_file_prompt")
                    if txt_file_prompt and self.do_diffusion:
                        print(f"New text file prompt received: {txt_file_prompt}")
                        self.transition_start_time = time.time()
                        self.embeds_source = self.em.clone_embeddings(self.embeds)

                        self.embeds_target = self.em.encode_prompt(txt_file_prompt)
                        self.fract_blend_embeds = 0.0

                        # permuted_prompt = self.permute_prompt(txt_file_prompt)
                        # print(f"Permuted prompt: {permuted_prompt}")

                        # self.embeds = self.em.encode_prompt(permuted_prompt)
                        # self.de_img.set_embeddings(self.embeds)

                    # Removed dynamic processor related processing.

                print(f"Received payload: {payload}")

                if self.bounce:
                    self.fps_tracker.start_segment("Bounce Mode")
                    img = recv_compressed(client_sock)
                    if img is None:
                        break
                    send_compressed(client_sock, img, quality=90)
                    continue

                print("Received payload")

                self.fps_tracker.start_segment("Receive Image")
                print("Waiting for compressed image from client...")
                img_cam = recv_compressed(client_sock)
                if img_cam is None:
                    print("Client disconnected during image receive")
                    break
                if not isinstance(img_cam, np.ndarray):
                    print(f"Invalid image received: {type(img_cam)}")
                    continue

                print(f"Received image! {type(img_cam)}")

                self.fps_tracker.start_segment("Input Image Processing")
                self.input_image_processor.set_human_seg(payload.get("do_human_seg", True))
                self.input_image_processor.set_resizing_factor_humanseg(0.4)
                self.input_image_processor.set_blur(payload.get("do_blur", False))
                self.input_image_processor.set_brightness(payload.get("brightness", 1.0))
                self.input_image_processor.set_infrared_colorize(payload.get("do_infrared_colorize", False))
                img_proc, human_seg_mask = self.input_image_processor.process(img_cam.copy())

                print("Processed image", img_proc.shape)
                # os.makedirs("output", exist_ok=True)
                # cv2.imwrite("output/processed_image.png", cv2.cvtColor(img_proc, cv2.COLOR_RGB2BGR))
                
                if not payload.get("do_human_seg", True):
                    human_seg_mask = np.ones_like(img_proc).astype(np.float32) / 255

                self.fps_tracker.start_segment("Acid Processing")
                self.acid_processor.set_acid_strength(payload.get("acid_strength", 0.11))
                self.acid_processor.set_coef_noise(payload.get("coef_noise", 0.15))
                self.acid_processor.set_acid_tracers(payload.get("do_acid_tracers", True))
                self.acid_processor.set_acid_strength_foreground(payload.get("acid_strength_foreground", 0.11))
                self.acid_processor.set_zoom_factor(payload.get("zoom_factor", 1.0))
                self.acid_processor.set_x_shift(payload.get("x_shift", 0))
                self.acid_processor.set_y_shift(payload.get("y_shift", 0))
                self.acid_processor.set_do_acid_wobblers(payload.get("do_acid_wobblers", False))
                self.acid_processor.set_color_matching(payload.get("color_matching", 0.5))

                img_acid = self.acid_processor.process(img_proc, human_seg_mask)

                self.fps_tracker.start_segment("Diffusion")

                # This is the image that will be diffused.
                self.de_img.set_input_image(img_acid)

                kwargs_override = {}

                img_diffusion = np.array(self.de_img.generate())
                self.acid_processor.update(img_diffusion)

                # Server no longer applies any postprocessing; just send the diffusion result.
                self.fps_tracker.start_segment("Send Result")
                print("DIFFUSION RESULT", img_diffusion.shape)
                send_compressed(client_sock, img_diffusion, quality=90)
                print("Sent result")

                self.fps_tracker.print_fps()

            except Exception as e:
                print("Error handling client:", e)
                import traceback
                traceback.print_exc()
                client_sock.close()
                break

        client_sock.close()

    def serve_forever(self):
        """Main loop to accept and serve clients."""
        try:
            while True:
                client_sock, addr = self.server_socket.accept()
                self.handle_client(client_sock, addr)
        except KeyboardInterrupt:
            print("Shutting down server...")
        finally:
            self.server_socket.close()

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

    if role == "server":
        # To test pure network latency, enable bounce mode by setting bounce=True.
        server = SubmersionServer(bounce=False, do_compile=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Server shutting down.")
    else:
        print("Invalid mode. Use 'server' or 'client'.")
        print("Invalid mode. Use 'server' or 'client'.")
        print("Invalid mode. Use 'server' or 'client'.")
