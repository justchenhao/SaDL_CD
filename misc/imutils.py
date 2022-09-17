import numpy as np
import cv2
from PIL import Image
import PIL


def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(np.array(image_numpy,dtype=np.uint8))
    image_pil.save(image_path)


def im2arr(img_path, mode=1, dtype=np.uint8):
    """
    :param img_path:
    :param mode:
    :return: numpy.ndarray, shape: H*W*C
    """
    if mode==1:
        img = PIL.Image.open(img_path)
        arr = np.asarray(img, dtype=dtype)
    elif mode==2:
        import skimage
        arr = skimage.io.imread(img_path)
        arr = arr.astype(dtype)
    elif mode == 4:
        import cv2 as cv
        arr = cv.imread(img_path, flags=cv2.IMREAD_UNCHANGED)
        arr = arr.astype(dtype)
    else:
        import tifffile
        arr = tifffile.imread(img_path)
        if arr.ndim == 3:
            a, b, c = arr.shape
            if a < b and a < c:  # 当arr为C*H*W时，需要交换通道顺序
                arr = arr.transpose([1, 2, 0])
    # print('shape: ', arr.shape, 'dytpe: ',arr.dtype)
    return arr


def mask_colorize(label_mask):
    """
    :param label_mask: mask (np.ndarray): (M, N), uint8
    :return: color label: (M, N, 3), uint8
    """
    assert isinstance(label_mask, np.ndarray)
    assert label_mask.ndim == 2
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.float)
    r = label_mask % 6
    g = (label_mask % 36) // 6
    b = label_mask // 36
    # 归一化到[0-1]
    rgb[:, :, 0] = r / 6
    rgb[:, :, 1] = g / 6
    rgb[:, :, 2] = b / 6
    rgb = np.array(rgb * 255, dtype=np.uint8)
    return rgb


def apply_colormap(norm_map, colormode='jet'):
    """
    输入：归一化的图，{[0,1]}^(H*W)；
    输出：上色后的图。
    """
    import cv2
    assert norm_map.ndim == 2

    if colormode == 'jet':
        colormap = cv2.COLORMAP_JET
    elif colormode == 'twilight':
        colormap = cv2.COLORMAP_TWILIGHT
    elif colormode == 'rainbow':
        colormap = cv2.COLORMAP_RAINBOW
    else:
        raise NotImplementedError

    norm_map_color = cv2.applyColorMap((norm_map * 255).astype(np.uint8),
                                       colormap=colormap)

    norm_map_color = norm_map_color[..., ::-1]

    return norm_map_color


