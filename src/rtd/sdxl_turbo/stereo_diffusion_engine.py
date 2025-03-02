class StereoDiffusionEngine(DiffusionEngine):
    """
    The StereoDiffusionEngine class is a subclass of DiffusionEngine that handles stereo images.
    It provides methods to set the stereo image flag, initialize the image for stereo processing, 
    and generate images for both left and right eyes.

    Attributes:
        do_stereo_image (bool): A flag to determine whether to use stereo image processing.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_stereo_image(self, do_stereo_image=False):
        """
        This method sets a boolean flag that indicates whether the diffusion has to be applied to left/right stereo image coming from VR.
        """
        self.do_stereo_image = do_stereo_image

    def set_input_image(self, image_init):
        """
        Sets the input image, resizing if necessary. If stereo image processing is enabled, it splits the image into left and right eye images.
        """
        if not isinstance(image_init, Image.Image):
            if image_init.dtype != np.uint8:
                image_init = np.round(image_init)
                image_init = np.clip(image_init, 0, 255)
                image_init = image_init.astype(np.uint8)
            image_init = Image.fromarray(image_init)
        
        if self.do_stereo_image:
            # the left/right eye images are stacked vertically
            sz = image_init.size
            img_left_eye = image_init.crop((0, 0, sz[0], sz[1]//2))
            img_right_eye = image_init.crop((0, sz[1]//2, sz[0], sz[1]))
            
            image_init = []
            for img_eye in [img_left_eye, img_right_eye]:
                width, height = img_eye.size
                if height != self.height_diffusion or width != self.width_diffusion:
                    img_eye = lt.resize(img_eye, size=(self.height_diffusion, self.width_diffusion))
                image_init.append(img_eye)
        else:
            width, height = image_init.size
            if height != self.height_diffusion or width != self.width_diffusion:
                image_init = lt.resize(image_init, size=(self.height_diffusion, self.width_diffusion))
        self.image_init = image_init

    def generate(self, kwargs_override=None, cross_attention_kwargs_override=None):
        """
        Generate an image using the current settings of the StereoDiffusionEngine.

        Args:
            kwargs_override (dict, optional): A dictionary of arguments to override the current settings of the StereoDiffusionEngine.
            cross_attention_kwargs_override (dict, optional): A dictionary of arguments to override the current settings of the cross_attention_kwargs.

        Returns:
            img_diffusion (torch.Tensor): The generated image.

        Raises:
            AssertionError: If embeddings are not set.
        """
        assert self.embeds is not None, "Embeddings not set! Call set_embeddings first."
        
        torch.manual_seed(self.seed)

        # First build the kwargs from the class attributes
        kwargs = self.build_kwargs(kwargs_override)

        # Then build the cross_attention_kwargs from the class attributes
        kwargs = self.build_cross_attention_kwargs(kwargs, cross_attention_kwargs_override)
        
        if self.do_diffusion:
            if self.do_stereo_image:
                img_diffusion = []                
                for img_eye in self.image_init:
                    kwargs['image'] = img_eye
                    img_eye_diffusion = self.pipe(**kwargs).images[0]
                    img_diffusion.append(np.array(img_eye_diffusion))
                img_diffusion = np.vstack(img_diffusion)
            else:
                img_diffusion = self.pipe(**kwargs).images[0]
        else:
            img_diffusion = None
    
        return img_diffusion