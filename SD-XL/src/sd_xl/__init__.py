import torch #Ejecutar opreciones matematicas del modelo
from diffusers import AutoPipelineForText2Image
# diffusers en una libreria especializada en mdoelos #Generativos
#
print("Cargando el modelo...")
#
modelo = AutoPipelineForText2Image.from_pretrained(
     "stabilityai/stable-diffusion-xl-base-1.0",
     torch_dtype=torch.float32 #float16 o float32
#     # torch.floatXX, indica la precision numerica
#     # con que s ecargan los pesos (decimales, la preciison para generar la imagen)
)

modelo = modelo.to("cpu")

prompt = input("Escribele el prompt de la imagenque quieres")
#negative _prompt = "worst quality, low resolution, blurry, distorted, deformed body, bad anatomy, deformed limbs, badly drawn hands, extra fingers, missing fingers, text, watermark, signature, logo, compression artifacts"

print("Gnerando imagen...")

imagen = modelo(
     prompt=prompt, #Instrucciones o detalle para crear la imagen
#     negative_prompt=negative_-prompt,
     num_interference_steps=25, #Las veces que va a tratar def
     guidance_scale=7.0,
     height=1024,
     width=1024

).images[0]

imagen.save("imagen.png")
#
print("Imagen guardada.")
# PS C:\Users\laura\Documents\sd-turbo-geni\src\sd_turbo_geni>
