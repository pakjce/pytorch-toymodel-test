import base64
from io import BytesIO
from PIL import Image
from matplotlib.figure import Figure


def image_to_base64(image: Image.Image) -> str:
    io = BytesIO()
    image.save(io, format='JPEG')
    io.seek(0)
    image_bytes = io.read()
    base64_str = 'data:image/jpeg;base64,'
    encoded = base64.b64encode(image_bytes).decode('utf-8')
    base64_str += encoded
    return base64_str


def figure_to_base64(plt: Figure):
    buf = BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image_bytes = buf.read()
    base64_str = 'data:image/jpeg;base64,'
    encoded = base64.b64encode(image_bytes).decode('utf-8')
    base64_str += encoded
    return base64_str
