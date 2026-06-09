import os
from math import isclose
from typing import Callable

import napari
import numpy as np
import pytest
from cellpose_napari._dock_widget import _V4

PLUGIN_NAME = "cellpose-napari"
WIDGET_NAME = "cellpose"

@pytest.fixture(autouse=True)
def force_cpu_on_CI(monkeypatch):
    # https://github.com/pytorch/pytorch/issues/75912
    if os.getenv('CI'):
        monkeypatch.setattr("cellpose.core.use_gpu", lambda *_, **__: False)


@pytest.fixture
def viewer_widget(make_napari_viewer: Callable[..., napari.Viewer]):
    viewer = make_napari_viewer()
    _, widget = viewer.window.add_plugin_dock_widget(
        plugin_name=PLUGIN_NAME, widget_name=WIDGET_NAME
    )
    return viewer, widget

def test_basic_function(qtbot, viewer_widget):
    viewer, widget = viewer_widget
    assert len(viewer.window.dock_widgets) == 1

    viewer.open_sample(PLUGIN_NAME, 'rgb_2D')

    if not _V4:
        widget.model_type.value = "cyto3"
    widget()  # run segmentation with all default parameters

    def check_widget():
        assert widget.cellpose_layers

    qtbot.waitUntil(check_widget, timeout=360_000)
    assert len(viewer.layers) == 5
    assert "cp_masks" in viewer.layers[-1].name
    # Slightly different results between cyto3 and cellpose-SAM
    if _V4:
        assert viewer.layers[-1].data.max() == 37
    else:
        assert viewer.layers[-1].data.max() == 41

def test_segment_2D_time(qtbot, viewer_widget, monkeypatch):
    viewer, widget = viewer_widget

    # synthetic 2D + time stack (T, Y, X)
    T, Y, X = 3, 96, 96
    rng = np.random.default_rng(0)
    stack = rng.integers(0, 255, size=(T, Y, X), dtype=np.uint8)

    # Fake the cellpose model so the test exercises only the stacking/UI logic
    # (no GPU, no weight download, no real inference — which otherwise hangs the worker).
    class _FakeCP:
        def __init__(self, *args, **kwargs):
            pass

        def eval(self, image, **kwargs):
            arr = np.asarray(image)
            t, y, x = arr.shape[0], arr.shape[1], arr.shape[2]
            masks = np.zeros((t, y, x), dtype=np.uint16)
            masks[:, 10:20, 10:20] = 1  # one labelled blob per frame
            # Return a short flows list (< 4 elements). 2D+time discards flows, so it must
            # not touch flows[3] — regression guard for the per-slice restructuring crash.
            flows = [
                np.zeros((t, y, x, 3), dtype=np.uint8),
                np.zeros((2, t, y, x), dtype=np.float32),
            ]
            return masks, flows, None

    monkeypatch.setattr("cellpose.models.CellposeModel", _FakeCP)

    layer = viewer.add_image(stack, name='movie')
    widget.image_layer.value = layer

    # enabling 2D+time must disable 3D processing (mutually exclusive)
    widget.segment_2D_time.value = True
    assert widget.process_3D.value == False

    if not _V4:
        widget.model_type.value = "cyto3"
    widget()

    def check_widget():
        assert widget.cellpose_layers

    qtbot.waitUntil(check_widget, timeout=60_000)

    # this run produced exactly one layer: a masks stack shaped like the input,
    # so the time slider scrubs masks and image together. No outlines/flows/cellprob.
    this_run = widget.cellpose_layers[-1]
    assert len(this_run) == 1
    assert 'cp_masks' in this_run[0].name
    assert this_run[0].data.shape == stack.shape
    # recompute-masks stays disabled (no flows retained)
    assert widget.compute_masks_button.enabled == False


@pytest.mark.skipif(_V4, reason="diameter estimation not available in cellpose v4")
def test_compute_diameter(qtbot, viewer_widget):
    viewer, widget = viewer_widget
    viewer.open_sample(PLUGIN_NAME, 'rgb_2D')

    assert widget.diameter.value == "30"
    with qtbot.waitSignal(widget.diameter.changed, timeout=60_000) as blocker:
        widget.compute_diameter_button.changed(None)

    # local on Windows with CPU/GPU 46.0
    assert isclose(float(widget.diameter.value), 46, abs_tol=0.3)


def test_widget_defaults(viewer_widget):
    _, widget = viewer_widget
    assert widget.diameter.value == "30"
    assert widget.cellprob_threshold.value == pytest.approx(0.0)
    assert widget.flow_threshold.value == pytest.approx(0.4)
    assert widget.min_size.value == 15
    assert widget.resample_dynamics.value == False
    assert widget.process_3D.value == False
    assert widget.segment_2D_time.value == False
    assert widget.stitch_threshold_3D.value == "0"
    assert widget.clear_previous_segmentations.value == True
    assert widget.compute_masks_button.enabled == False
    if _V4:
        assert widget.z_axis.native.isHidden()
        assert widget.channel_axis.native.isHidden()


def test_dimensionality_detection(viewer_widget):
    viewer, widget = viewer_widget

    layer_3d = viewer.add_image(np.zeros((5, 64, 64)), name='stack')
    widget.image_layer.value = layer_3d
    assert widget.process_3D.value == True
    if _V4:
        # isVisible() requires all parents to be visible; isHidden() checks the widget's own flag
        assert not widget.z_axis.native.isHidden()
        assert not widget.channel_axis.native.isHidden()

    layer_2d = viewer.add_image(np.zeros((64, 64)), name='flat')
    widget.image_layer.value = layer_2d
    assert widget.process_3D.value == False
    if _V4:
        assert widget.z_axis.native.isHidden()
        assert widget.channel_axis.native.isHidden()


def test_diameter_from_shape(viewer_widget):
    viewer, widget = viewer_widget

    # 30×30 square: ptp=[30,30], sum=60 → diam = (60/2) * (27/30) = 27.0
    rect = np.array([[0, 0], [0, 30], [30, 30], [30, 0]], dtype=float)
    shape_layer = viewer.add_shapes([rect], shape_type='rectangle')
    widget.shape_layer.value = shape_layer

    widget.compute_diameter_shape.changed(None)

    assert float(widget.diameter.value) == pytest.approx(27.0)
