import os
from pathlib import Path
from math import isclose
from typing import Callable

import napari
import pytest
import torch # for ubuntu tests on CI, see https://github.com/pytorch/pytorch/issues/75912

from cellpose_napari._dock_widget import _V4

PLUGIN_NAME = "cellpose-napari"
WIDGET_NAME = "cellpose"

@pytest.fixture(autouse=True)
def patch_mps_on_CI(monkeypatch):
    # https://github.com/actions/runner-images/issues/9918
    if os.getenv('CI'):
        monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
        monkeypatch.setattr("cellpose.core.assign_device", lambda **kwargs: (torch.device("cpu"), False))


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

    qtbot.waitUntil(check_widget, timeout=60_000)
    assert len(viewer.layers) == 5
    assert "cp_masks" in viewer.layers[-1].name
    # Slightly different results between cyto3 and cellpose-SAM
    if _V4:
        assert viewer.layers[-1].data.max() == 37
    else:
        assert viewer.layers[-1].data.max() == 40

@pytest.mark.skipif(_V4, reason="diameter estimation not available in cellpose v4")
def test_compute_diameter(qtbot, viewer_widget):
    viewer, widget = viewer_widget
    viewer.open_sample(PLUGIN_NAME, 'rgb_2D')

    assert widget.diameter.value == "30"
    with qtbot.waitSignal(widget.diameter.changed, timeout=60_000) as blocker:
        widget.compute_diameter_button.changed(None)

    # local on Windows with CPU/GPU 46.0
    assert isclose(float(widget.diameter.value), 46, abs_tol=0.3)

def test_3D_segmentation(qtbot,  viewer_widget):
    viewer, widget = viewer_widget
    assert widget.process_3D.value == False
    viewer.open_sample(PLUGIN_NAME, 'rgb_3D')
    # viewer.layers[0].data = viewer.layers[0].data[:20]
    assert widget.process_3D.value == True

    if not _V4:
        widget.model_type.value = "cyto3"
    widget()  # run segmentation with all default parameters

    def check_widget():
        assert widget.cellpose_layers

    qtbot.waitUntil(check_widget, timeout=120_000)
    assert len(viewer.layers) == 5
    assert "cp_masks" in viewer.layers[-1].name
    assert viewer.layers[-1].data.max() == 7
