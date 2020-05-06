from flask import current_app, Blueprint, render_template, request
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import numpy as np

import config
from os.path import join
from utils import image_to_base64, get_pil_image_from_dataset, figure_to_base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from prepare_model import MNISTNet


bp_inference = Blueprint('inference', __name__, url_prefix='/inference')


@bp_inference.route('/')
def inference():
    testId = request.args.get('testId', None)
    if testId is None:
        return "ERROR: Invalid testId Parameter", 400

    dataset = MNIST(
        config.DATA_DIR, train=False
    )

    test_id = int(testId)
    if test_id < 0 or test_id >= dataset.data.shape[0]:
        return "ERROR: Out of Range testId Parameter", 400

    test_img = image_to_base64(
        get_pil_image_from_dataset(dataset, test_id)
    )

    device = torch.device('cpu')
    model = MNISTNet()
    model.load_state_dict(
        torch.load(join(config.MODEL_DIR, 'mnist_cnn.pt'), map_location=device)
    )
    model.eval()

    normalized_data = MNIST(
        config.DATA_DIR, train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    with torch.no_grad():
        data = normalized_data.data[test_id, :, :]
        data = data.type(torch.float32)
        data = data.unsqueeze(0).unsqueeze(0)
        target = normalized_data.targets[test_id]

        data, target = data.to(device), target.to(device)

        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

    output_val = np.exp(output.numpy()).squeeze()
    pred_idx = int(pred.numpy()[0][0])
    pred_score = output_val[pred_idx]
    gt_label = dataset.classes[int(dataset.test_labels[test_id])]
    pred_label = dataset.classes[pred_idx]

    fig = Figure(figsize=(10, 10), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    x = np.arange(0, 10, 1)
    ax.set_xticks(x)
    # Set ticks labels for x-axis
    ax.set_xticklabels(dataset.classes, rotation='vertical', fontsize=18)
    # ax = fig.add_axes([.1, .1, .8, .8])
    ax.plot(output_val)
    # ax.title('Predictions')

    inf_img = figure_to_base64(fig)


    return render_template(
        'inference.html',
        test_id=test_id,
        test_img=test_img,
        inf_img=inf_img,
        gt_label=gt_label,
        pred_label=pred_label
    )
