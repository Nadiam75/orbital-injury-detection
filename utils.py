import numpy as np
import cv2
from prettytable import PrettyTable
import torch.nn as nn
import torch

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None


def crop_image(image: np.ndarray) -> np.ndarray:
    """
    Crops an image based on its minimum pixel value (assumed background)
    and resizes it to 64x128.

    Args:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        np.ndarray: The cropped and resized image.
    """
    mask = image == image.min()
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)
    cropped_image = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    return cv2.resize(cropped_image, (64, 128))


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizes a NumPy array image to a [0, 1] range.

    Args:
        image (np.ndarray): The input image array.

    Returns:
        np.ndarray: The normalized image array.
    """
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def apply_windowing(image: np.ndarray, window_params: tuple, d2: tuple = None, d3: tuple = None) -> np.ndarray:
    """
    Applies windowing to an image based on provided window parameters.
    This function can apply up to three different window levels and widths
    and sum their results.

    Args:
        image (np.ndarray): The input image array.
        window_params (tuple): A tuple (min_val, max_val) for the first window.
        d2 (tuple, optional): A tuple (min_val, max_val) for the second window. Defaults to None.
        d3 (tuple, optional): A tuple (min_val, max_val) for the third window. Defaults to None.

    Returns:
        np.ndarray: The windowed image, which can be a sum of multiple windowed images.
    """
    # Initialize images for windowing
    windowed_image1 = image.copy()
    windowed_image2 = image.copy()
    windowed_image3 = image.copy()

    # Apply first window
    min_val1, max_val1 = window_params
    windowed_image1[windowed_image1 < min_val1] = min_val1
    windowed_image1[windowed_image1 > max_val1] = max_val1

    result_image = windowed_image1

    # Apply second window if parameters are provided
    if d2:
        min_val2, max_val2 = d2
        windowed_image2[windowed_image2 < min_val2] = min_val2
        windowed_image2[windowed_image2 > max_val2] = max_val2
        result_image += windowed_image2

    # Apply third window if parameters are provided
    if d3:
        min_val3, max_val3 = d3
        windowed_image3[windowed_image3 < min_val3] = min_val3
        windowed_image3[windowed_image3 > max_val3] = max_val3
        result_image += windowed_image3

    return result_image


def count_model_parameters(model: nn.Module) -> int:
    """
    Counts and prints the number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        int: The total number of trainable parameters.
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    # print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def create_video_from_numpy(numpy_array: np.ndarray, save_path: str, fps: int = 8) -> None:
    """
    Creates a video from a 3D NumPy array (channels, height, width).
    """
    height, width = numpy_array.shape[1], numpy_array.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height), isColor=False)

    for i in range(numpy_array.shape[0]):
        frame = (numpy_array[i] * 255).astype(np.uint8)
        video_writer.write(frame)

    video_writer.release()


def get_conv_layer(model: nn.Module, conv_layer_name: str) -> nn.Module:
    """
    Retrieves a module from a model by its dotted name as returned by model.named_modules().
    """
    for name, layer in model.named_modules():
        if name == conv_layer_name:
            return layer
    raise ValueError(f"Layer '{conv_layer_name}' not found in the model.")


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculates binary accuracy (%) from logits and labels.
    """
    predicted = torch.sigmoid(outputs).round()
    return (predicted == labels).sum().item() / labels.size(0) * 100


def get_module_by_name(model: nn.Module, dotted: str) -> nn.Module:
    """
    Resolve a submodule by dotted path (e.g., 'ax_stem1.0').
    """
    m = model
    for name in dotted.split('.'):
        if not hasattr(m, name):
            raise AttributeError(f"Missing submodule '{name}' in path '{dotted}'")
        m = getattr(m, name)
    return m


def normalize01(a: np.ndarray) -> np.ndarray:
    """
    Min-max normalize an array to [0, 1]; returns zeros if constant.
    """
    a = a.astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx > mn:
        return (a - mn) / (mx - mn)
    return np.zeros_like(a, dtype=np.float32)


def pick_slice(vol: torch.Tensor, slice_idx=None) -> np.ndarray:
    """
    Pick a single slice from a tensor for visualization.
    Accepts [1,S,H,W] or [S,H,W] (or [H,W]) after squeezing.
    """
    v = vol.squeeze(0).detach().cpu().numpy()
    if v.ndim == 3:  # [S,H,W]
        S = v.shape[0]
        idx = S // 2 if slice_idx is None else max(0, min(S - 1, slice_idx))
        return v[idx]
    return v  # [H,W]


def ensure_same_hw(cam: np.ndarray, base: np.ndarray) -> np.ndarray:
    if cam.shape != base.shape:
        cam = cv2.resize(cam, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
    return cam


def add_img_subplot(fig, row: int, col: int, img01: np.ndarray, title: str) -> None:
    if go is None:
        raise RuntimeError("plotly is required for add_img_subplot")
    fig.add_trace(
        go.Heatmap(z=img01, colorscale='Gray', showscale=False),
        row=row,
        col=col,
    )
    fig.update_xaxes(visible=False, row=row, col=col)
    fig.update_yaxes(visible=False, autorange='reversed', row=row, col=col)
    fig.layout.annotations[(row - 1) * 2 + (col - 1)].update(text=title)


def add_overlay_subplot(
    fig,
    row: int,
    col: int,
    base01: np.ndarray,
    cam01: np.ndarray,
    title: str,
    alpha: float = 0.6,
    show_colorbar: bool = False,
) -> None:
    if go is None:
        raise RuntimeError("plotly is required for add_overlay_subplot")
    fig.add_trace(
        go.Heatmap(z=base01, colorscale='Gray', showscale=False),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Heatmap(
            z=cam01,
            colorscale='Turbo',
            opacity=alpha,
            showscale=show_colorbar,
            colorbar=dict(title="CAM", len=0.8) if show_colorbar else None,
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(visible=False, row=row, col=col)
    fig.update_yaxes(visible=False, autorange='reversed', row=row, col=col)
    fig.layout.annotations[(row - 1) * 2 + (col - 1)].update(text=title)