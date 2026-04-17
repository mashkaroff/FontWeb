import os
import io
import base64
from PIL import Image, ImageDraw, ImageFont

def generate_font(font_path):
    image = Image.new("RGB",  (800, 100), color = (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, size=25)
    text = "Съешь ещё этих французских булок, да выпей чаю"
    color = (0, 0, 0)
    draw.text((20, 20), text, fill=color, font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return img_str