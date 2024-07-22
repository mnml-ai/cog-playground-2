#@title generator class
#writing this class seperate from cog :: BECAUSE I HATE DOCKER :: And to make it work in colab and envs where docker is not available

import torch
import os
from PIL import Image, ImageEnhance

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetImg2ImgPipeline
)
from compel import Compel
from diffusers.models import AutoencoderKL
from transformers import CLIPVisionModelWithProjection
from utils import *
import time
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from torch import nn
import numpy as np

class Generator:
    def __init__(self, sd_path= "stablediffusionapi/majicmix-v7", vae_path= None, load_ip_adapter=False, load_controlnets={}, use_compel= False, ip_image_encoder= "weights/image_encoder", ip_weight="weights/ip-adapter_sd15.bin" ):

        self.use_compel = use_compel
        self.load_ip_adapter = load_ip_adapter
        self.ip_weight = ip_weight
        self.controlnets = {}
        self.preprocessors = {}
        self.detectors = {}

        if vae_path:
            vae = AutoencoderKL.from_pretrained(vae_path)

        if load_ip_adapter:
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(ip_image_encoder, local_files_only=True,).to("cuda", dtype=torch.float16)

        self.pipe = StableDiffusionPipeline.from_pretrained(
            sd_path, torch_dtype=torch.float16,
            # local_files_only=True,
            vae= vae if vae_path else None
        )
        # self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        # self.pipe.to("cuda")

        if load_controlnets:
            for name in load_controlnets:
                print("loading controlnets...")
                model= AUX_IDS[name]
                self.controlnets[name] = ControlNetModel.from_pretrained(
                    model["path"],
                    torch_dtype=torch.float16,
                    # local_files_only=True,
                ).to("cuda")
                print("loading controlnet detectors..")
                self.detectors[name] = model['detector']()

        if self.use_compel:
            self.compel_proc = Compel(tokenizer=self.pipe.tokenizer, text_encoder=self.pipe.text_encoder)

        #load clip
        self.clip_seg_processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.clip_seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

        #LOAD LORAS
        self.pipe.load_lora_weights("dsgnrai/lora", weight_name="more_details.safetensors", adapter_name="more_details")
        self.pipe.load_lora_weights("dsgnrai/lora", weight_name="epi_noiseoffset2.safetensors", adapter_name="epi_noiseoffset2")
        self.pipe.load_lora_weights("dsgnrai/lora", weight_name="color_temperature_slider_v1.safetensors", adapter_name="color_temperature_slider_v1")
        self.pipe.load_lora_weights("dsgnrai/lora", weight_name="add_detail.safetensors", adapter_name="add_detail")
        self.pipe.load_lora_weights("dsgnrai/lora", weight_name="FilmVelvia3.safetensors", adapter_name="FilmVelvia3")
        self.pipe.load_lora_weights("dsgnrai/lora", weight_name="mp_v1.safetensors", adapter_name="mp_v1")
        self.pipe.load_lora_weights("dsgnrai/lora", weight_name="id_v1.safetensors", adapter_name="id_v1")
        self.pipe.load_lora_weights("dsgnrai/lora", weight_name="ex_v1.safetensors", adapter_name="ex_v1")
        self.pipe.load_lora_weights("dsgnrai/lora", weight_name="SDXLrender_v2.0.safetensors", adapter_name="SDXLrender_v2")

        #load textual inversions
        self.pipe.load_textual_inversion("dsgnrai/negative-embeddings", weight_name="FastNegativeV2.pt", token="FastNegativeV2")
        self.pipe.load_textual_inversion("dsgnrai/negative-embeddings", weight_name="boring_e621_v4.pt", token="boring_e621_v4")
        self.pipe.load_textual_inversion("dsgnrai/negative-embeddings", weight_name="verybadimagenegative_v1.3.pt", token="verybadimagenegative_v1")
        self.pipe.load_textual_inversion("dsgnrai/negative-embeddings", weight_name="JuggernautNegative-neg.pt", token="JuggernautNegative-neg")

        self.pipe.to("cuda", torch.float16)

    def convert_image(self, image):
        #converts black pixels into transparent and white pixels to black
        grayscale_image = image.convert("L")
        new_image = Image.new("RGBA", grayscale_image.size)

        for x in range(grayscale_image.width):
            for y in range(grayscale_image.height):
                intensity = grayscale_image.getpixel((x, y))
                if intensity == 255:
                    new_image.putpixel((x, y), (0, 0, 0, 255))
                elif intensity == 0:
                    new_image.putpixel((x, y), (0, 0, 0, 0))

        return new_image

    def segment_image(self, texts, image, negative= False):

        #dont know why when there is only one text nn.functional.interpolate gives error, so
        if len(texts)==1:
            texts= [texts[0], texts[0]]

        images = [image] * len(texts)

        inputs = self.clip_seg_processor(text=texts, images=images, padding=True, return_tensors="pt")
        outputs = self.clip_seg_model(**inputs)

        preds = nn.functional.interpolate(
            outputs.logits.unsqueeze(1),
            size=(images[0].size[1], images[0].size[0]),
            mode="bilinear"
        )

        # Create an empty blended image with the same size as the first prediction
        blended_image = Image.fromarray((torch.sigmoid(preds[0][0]).detach().numpy() * 255).astype(np.uint8))

        # Iterate over the remaining predictions and blend them with the existing blended image
        for pred in preds[1:]:
            current_image = Image.fromarray((torch.sigmoid(pred[0]).detach().numpy() * 255).astype(np.uint8))
            blended_image = Image.blend(blended_image, current_image, alpha=0.5)

        # Enhance the contrast and brightness of the blended image
        enhancer = ImageEnhance.Contrast(blended_image)
        blended_image = enhancer.enhance(2.0)  # Adjust the factor as needed

        enhancer = ImageEnhance.Brightness(blended_image)
        blended_image = enhancer.enhance(2.5)  # Adjust the factor as needed
        
        if negative:
            blended_image = self.convert_image(blended_image)

        return blended_image


    def build_pipe(
            self, inputs, max_width, max_height, guess_mode=False, use_ip_adapter= False, img2img=None, img2img_strength= 0.8
        ):
        print("using ip adapter::", use_ip_adapter)
        if use_ip_adapter:
            from Diffusers_IPAdapter.ip_adapter.ip_adapter import IPAdapter
        control_nets = []
        processed_control_images = []
        conditioning_scales = []
        w, h = max_width, max_height
        inpainting = False
        #image and mask for inpainting
        mask= None
        init_image= None
        got_size= False
        img2img_image= None
        for name, [image, conditioning_scale, mask_image,  text_for_auto_mask, negative_text_for_auto_mask] in inputs.items():
            if image is None:
                continue
            # print(name)
            if not isinstance(image, Image.Image):
                image = Image.open(image)
            if not got_size:
                image= resize_image(image, max_width, max_height)
                w, h= image.size
                got_size= True
            else:
                image= image.resize((w,h))

            if name=="inpainting" and (mask_image or text_for_auto_mask or negative_text_for_auto_mask) :
                inpainting = True
                if text_for_auto_mask:
                    print("generating mask")
                    ti = time.time()
                    mask = self.segment_image(text_for_auto_mask, image).resize((w,h))
                    print(f"Time taken to generate mask-- : {time.time() - ti:.2f} seconds")
                    ti = time.time()
                    if negative_text_for_auto_mask:
                        n_mask= self.segment_image(negative_text_for_auto_mask, image, negative=True).resize((w,h))
                        mask = Image.alpha_composite(mask.convert("RGBA"), n_mask)
                        print(f"Time taken to generate negative mask-- : {time.time() - ti:.2f} seconds")
                    print(image.size, 'img size/// mask -', mask.size )
                    img = AUX_IDS[name]["preprocessor"](self, image, mask)
                else:
                    mask_image= Image.open(mask_image)
                    mask= mask_image.resize((w,h))
                    img= AUX_IDS[name]["preprocessor"](self, image, mask)
                init_image= image
                inpaint_strength= conditioning_scale
                inpaint_img= img
            else:
                img= AUX_IDS[name]["preprocessor"](self, image)
                img= img.resize((w,h))

            control_nets.append(self.controlnets[name])
            processed_control_images.append(img)
            conditioning_scales.append(conditioning_scale)

        if img2img:
            print('image 2 image', img2img)
            if not isinstance(img2img, Image.Image):
                img2img_image= Image.open(img2img)
            if not got_size:
                print("not got size")
                img2img_image= resize_image(img2img_image, max_width, max_height)
            else:
                try:
                    img2img_image= img2img_image.resize(w,h)
                except:
                    print("git error in resizing")
                    img2img_image= resize_image(img2img_image, max_width, max_height)

        ip= None
        if len(control_nets) == 0:
            pipe = self.pipe
            kwargs = {"width":max_width, "height": max_height}
            if use_ip_adapter:
                ip = IPAdapter(pipe, self.ip_weight, self.image_encoder, device="cuda")
        else:
            if inpainting:
                pipe = StableDiffusionControlNetInpaintPipeline(
                    vae=self.pipe.vae,
                    text_encoder=self.pipe.text_encoder,
                    tokenizer=self.pipe.tokenizer,
                    unet=self.pipe.unet,
                    scheduler=self.pipe.scheduler,
                    safety_checker=self.pipe.safety_checker,
                    feature_extractor=self.pipe.feature_extractor,
                    controlnet=control_nets,
                )
                if use_ip_adapter:
                    ip = IPAdapter(pipe, self.ip_weight, self.image_encoder, device="cuda")
                kwargs = {
                    "image": init_image,
                    "mask_image": mask,
                    "control_image": processed_control_images,
                    "controlnet_conditioning_scale": conditioning_scales,
                    "guess_mode": guess_mode,
                    "strength": inpaint_strength
                }
            elif img2img:
                pipe = StableDiffusionControlNetImg2ImgPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                safety_checker=self.pipe.safety_checker,
                feature_extractor=self.pipe.feature_extractor,
                controlnet=control_nets,
                )
                if use_ip_adapter:
                    ip = IPAdapter(pipe, self.ip_weight, self.image_encoder, device="cuda")
                kwargs = {
                    "image": img2img_image,
                    "control_image": processed_control_images,
                    "controlnet_conditioning_scale": conditioning_scales,
                    "guess_mode": guess_mode,
                    "strength":img2img_strength
                }
            else:
                pipe = StableDiffusionControlNetPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                safety_checker=self.pipe.safety_checker,
                feature_extractor=self.pipe.feature_extractor,
                controlnet=control_nets,
                )
                if use_ip_adapter:
                    ip = IPAdapter(pipe, self.ip_weight, self.image_encoder, device="cuda")
                kwargs = {
                    "image": processed_control_images,
                    "controlnet_conditioning_scale": conditioning_scales,
                    "guess_mode": guess_mode,
                }
                # print(kwargs, control_nets)
        t= time.time()
        # pipe.load_lora_weights("/content/more_details.safetensors")
        # print(f"Time taken to load lora: {time.time() - t:.2f} seconds")
        return pipe, kwargs, ip

    def predict(self, prompt="", lineart_image=None, lineart_conditioning_scale=1.0,
                scribble_image=None, scribble_conditioning_scale=1.0,
                tile_image=None, tile_conditioning_scale=1.0,
                brightness_image=None, brightness_conditioning_scale=1.0,
                inpainting_image=None, mask_image=None, inpainting_conditioning_scale=1.0,
                depth_conditioning_scale= 1.0, depth_image= None,
                mlsd_image= None, mlsd_conditioning_scale=1.0,
                canny_conditioning_scale= 1.0, canny_image= None,
                
                num_outputs=1, max_width=512, max_height=512,
                scheduler="DDIM", num_inference_steps=20, guidance_scale=7.0,
                seed=None, eta=0.0,
                negative_prompt="Longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                guess_mode=False, disable_safety_check=False,
                sorted_controlnets="tile, inpainting, lineart",
                ip_adapter_image=None, ip_adapter_weight=1.0,
                img2img=None, img2img_strength= 0.8, ip_ckpt='"ip-adapter_sd15.bin"',
                text_for_auto_mask=None, negative_text_for_auto_mask= None,

                add_more_detail_lora_scale= 0, detail_tweaker_lora_weight= 0, film_grain_lora_weight= 0, 
                epi_noise_offset_lora_weight=0, color_temprature_slider_lora_weight=0,
                mp_lora_weight=0, id_lora_weight=0, ex_v1_lora_weight=0, SDXLrender_v2_lora_weight=0,

                ):
        
        lora_weights=[]
        loras= []
        if add_more_detail_lora_scale!=0:
            lora_weights.append(add_more_detail_lora_scale)
            loras.append("more_details")
        if detail_tweaker_lora_weight!=0:
            lora_weights.append(detail_tweaker_lora_weight)
            loras.append("add_detail")
        if film_grain_lora_weight!=0:
            lora_weights.append(film_grain_lora_weight)
            loras.append("FilmVelvia3")
        if epi_noise_offset_lora_weight!=0:
            lora_weights.append(epi_noise_offset_lora_weight)
            loras.append("epi_noiseoffset2")
        if color_temprature_slider_lora_weight!=0:
            lora_weights.append(color_temprature_slider_lora_weight)
            loras.append("color_temperature_slider_v1")
        if mp_lora_weight!=0:
            lora_weights.append(mp_lora_weight)
            loras.append("mp_v1")
        if id_lora_weight!=0:
            lora_weights.append(id_lora_weight)
            loras.append("id_v1")
        if ex_v1_lora_weight!=0:
            lora_weights.append(ex_v1_lora_weight)
            loras.append("ex_v1")
        if SDXLrender_v2_lora_weight!=0:
            lora_weights.append(SDXLrender_v2_lora_weight)
            loras.append("SDXLrender_v2")


        t1= time.time()
        self.ip_weight= f"weights/{ip_ckpt}"

        if not disable_safety_check and 'nude' in prompt:
            raise Exception(
                f"NSFW content detected. try a different prompt."
            )
        #dont know why, if ip adapter image is not given, it produce green image- so quick fix for non-ip adapter generations - will it soon MAYBE
        if not ip_adapter_image:
            ip_adapter_image= 'example/cat.png'
            ip_adapter_weight= 0.0

        control_inputs= {
                "brightness": [brightness_image, brightness_conditioning_scale, None, None, None],
                "tile": [tile_image, tile_conditioning_scale, None, None, None],
                "lineart": [lineart_image, lineart_conditioning_scale, None, None, None],
                "inpainting": [inpainting_image, inpainting_conditioning_scale, mask_image, text_for_auto_mask, negative_text_for_auto_mask],
                "scribble": [scribble_image, scribble_conditioning_scale, None, None, None],
                "depth": [depth_image, depth_conditioning_scale, None, None, None],
                "mlsd": [mlsd_image, mlsd_conditioning_scale, None, None, None],
                "canny": [canny_image, canny_conditioning_scale, None, None, None],
            }
        sorted_control_inputs= sort_dict_by_string(sorted_controlnets, control_inputs)
        t2= time.time()
        print(f"Time taken until build pipe: {t2 - t1:.2f} seconds")
        pipe, kwargs, ip = self.build_pipe(
            sorted_control_inputs,
            max_width=max_width,
            max_height=max_height,
            guess_mode=guess_mode,
            use_ip_adapter= ip_adapter_image,
            img2img=img2img, img2img_strength= img2img_strength
        )
        t3= time.time()
        print(f"Time taken to build pipe: {t3 - t2:.2f} seconds")
        if scheduler=='DPMSolverMultistep':
            pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config, algorithm_type="sde-dpmsolver++")
        else:
            pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        t4= time.time()
        print(f"Time taken to apply scheduler-- : {t4 - t3:.2f} seconds")
        t5= time.time()
        print(f"Time taken to cuda-- : {t5 - t4:.2f} seconds")
        # pipe.enable_xformers_memory_efficient_attention()
        
        
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # generator = torch.Generator("cuda").manual_seed(seed)

        if disable_safety_check:
            pipe.safety_checker = None

        outputs= []
        for idx in range(num_outputs):
            this_seed = seed + idx
            generator = torch.Generator("cuda").manual_seed(seed)
            pipe.set_adapters(loras, adapter_weights=lora_weights)
            pipe.fuse_lora()
            if ip_adapter_image:
                t6= time.time()
                print(f"Time taken until ip -- : {t6 - t5:.2f} seconds")
                ip_image= Image.open(ip_adapter_image)
                prompt_embeds_, negative_prompt_embeds_ = ip.get_prompt_embeds(
                    ip_image,
                    p_embeds=self.compel_proc(prompt),
                    n_embeds=self.compel_proc(negative_prompt),
                    weight=[ip_adapter_weight]
                )
                t7= time.time()
                print(f"Time taken to load ip-- : {t7 - t6:.2f} seconds")
                output = pipe(
                    prompt_embeds= prompt_embeds_,
                    negative_prompt_embeds= negative_prompt_embeds_,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    num_images_per_prompt=1,
                    generator=generator,
                    # output_type="pil",
                    **kwargs,
                )
                t8 = time.time()
                print(f"Time taken to generate image-- : {t8 - t7:.2f} seconds")
            else:
                output = pipe(
                    prompt_embeds=self.compel_proc(prompt),
                    negative_prompt_embeds=self.compel_proc(negative_prompt),
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    num_images_per_prompt=1,
                    generator=generator,
                    # output_type="pil",
                    **kwargs,
                )
            pipe.unfuse_lora()
            if output.nsfw_content_detected and output.nsfw_content_detected[0]:
                continue
            outputs.append(output)
        t9= time.time()
        # print(f"Time taken after generating image: {t9 - t8:.2f} seconds", f"/// total time taken: {t9 - t1:.2f}")
        return outputs

