from torchvision.datasets import MNIST
from PIL import Image


def get_pil_image_from_dataset(dataset: MNIST, index: int) -> Image.Image:
    img = dataset.data[index]
    img = Image.fromarray(img.numpy(), mode='L')
    return img


if __name__ == '__main__':
    import config
    from os.path import join
    dataset = MNIST(
        config.DATA_DIR, train=False
    )

    tp = get_pil_image_from_dataset(dataset, 0)

    tp.save(join(config.DATA_DIR, 'test.jpg'))
    pass