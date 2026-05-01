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
