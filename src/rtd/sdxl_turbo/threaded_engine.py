class DiffusionEngineThreaded:
    DEFAULT_NUM_INFERENCE_STEPS = 2

    def __init__(self, diffusion_engine):
        self.diffusion_engine = diffusion_engine
        self.do_run = False
        self.num_inference_steps = self.DEFAULT_NUM_INFERENCE_STEPS
        self.embeds = None
        self.input_image = None
        self.latents = None
        self.strength = None
        self.decoder_embeds = None
        self.last_diffusion_img = None  # Initialize as None
        self.start_generation_thread()  # Start the thread in the init

    def start_generation_thread(self):
        generation_thread = threading.Thread(target=self._run_generation)
        generation_thread.start()

    def set_embeddings(self, embeds):
        self.embeds = embeds

    def set_input_image(self, input_image):
        self.input_image = input_image

    def set_latents(self, latents):
        self.latents = latents

    def set_decoder_embeddings(self, decoder_embeds):
        self.decoder_embeds = decoder_embeds

    def set_num_inference_steps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps

    def _run_generation(self):
        while True:
            if not self.do_run:
                time.sleep(0.02)
                continue
            if self.input_image is not None:
                input_image = self.input_image
            else:
                input_image = np.array(self.diffusion_engine.image_init)

            self.diffusion_engine.set_embeddings(self.embeds)
            if input_image is not None:
                self.diffusion_engine.set_input_image(input_image)
            if self.latents is not None:
                self.diffusion_engine.set_latents(self.latents)
            if self.decoder_embeds is not None:
                self.diffusion_engine.set_decoder_embeddings(self.decoder_embeds)
            if self.num_inference_steps is not None:
                self.diffusion_engine.set_num_inference_steps(int(self.num_inference_steps))
            if self.strength is not None:
                self.diffusion_engine.set_strength(self.strength)

            img = np.asarray(self.diffusion_engine.generate())
            self.last_diffusion_img = img
            # upd liveport...

            self.do_run = False

    def generate(self, embeds, input_image=None, latents=None, decoder_embeds=None, num_inference_steps=None, strength=None, do_run=True):
        self.do_run = do_run
        self.latents = latents
        self.decoder_embeds = decoder_embeds
        if num_inference_steps != self.num_inference_steps:
            self.num_inference_steps = num_inference_steps
            self.do_run = True
        if strength != self.strength:
            self.strength = np.clip(strength, 1/self.num_inference_steps, 1.0)
            self.do_run = True
        if self.embeds is None or not torch.equal(self.embeds[0], embeds[0]):
            self.embeds = embeds
            self.do_run = True
        if input_image is not None:
            input_image = np.array(input_image)  # Convert PIL image to numpy array
            if not np.array_equal(self.input_image, input_image):  # Check if different from self.input_image
                self.input_image = input_image
                self.do_run = True
        
        return self.last_diffusion_img