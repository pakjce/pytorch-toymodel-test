from flask import current_app, Blueprint, render_template
from torchvision.datasets import MNIST

import config
from os.path import join
from utils import image_to_base64, get_pil_image_from_dataset


bp_index = Blueprint('index', __name__, url_prefix='/')


@bp_index.route('/')
def index():

    dataset = MNIST(
        config.DATA_DIR, train=False
    )
    images = []
    for i in range(0, min(int(dataset.data.shape[0]), 300)):
        images.append(
            (i, image_to_base64(
                get_pil_image_from_dataset(dataset, i)
            ))
        )
    return render_template(
        'index.html',
        sample_images=images
    )
