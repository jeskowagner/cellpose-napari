"""
cellpose dock widget — supports cellpose v3 and v4
"""
from typing import Any
from importlib.metadata import version as _pkg_version

import numpy as np
import logging

logger = logging.getLogger(__name__)

_V4 = int(_pkg_version('cellpose').split('.')[0]) >= 4

cp_strings = ['_cp_masks_', '_cp_outlines_', '_cp_flows_', '_cp_cellprob_']

if not _V4:
    _CP_models = ["cyto3", "cyto2", "cyto", "nuclei", "tissuenet_cp3",
                  "livecell_cp3", "yeast_PhC_cp3", "yeast_BF_cp3", "bact_phase_cp3",
                  "bact_fluor_cp3", "deepbacs_cp3", "cyto2_cp3"]
    _main_channel_choices = [('average all channels', 0), ('0=red', 1), ('1=green', 2), ('2=blue', 3),
                              ('3', 4), ('4', 5), ('5', 6), ('6', 7), ('7', 8), ('8', 9)]
    _nuclear_channel_choices = [('none', 0), ('0=red', 1), ('1=green', 2), ('2=blue', 3),
                                 ('3', 4), ('4', 5), ('5', 6), ('6', 7), ('7', 8), ('8', 9)]


def widget_wrapper():
    from napari import Viewer
    from napari.layers import Image, Shapes
    from magicgui import magicgui
    from napari.qt.threading import thread_worker
    try:
        from torch import no_grad
    except ImportError:
        def no_grad():
            def _deco(func):
                return func
            return _deco

    @thread_worker
    @no_grad()
    def run_cellpose(image, diameter, resample, cellprob_threshold, flow_threshold,
                     min_size, do_3D, stitch_threshold,
                     model_type=None, custom_model=None, channels=None, channel_axis=None):
        from cellpose import models

        if _V4:
            CP = models.CellposeModel(gpu=True)
        elif model_type == 'custom':
            CP = models.CellposeModel(pretrained_model=custom_model, gpu=True)
        else:
            CP = models.CellposeModel(model_type=model_type, gpu=True)

        eval_kwargs = dict(
            diameter=diameter,
            resample=resample,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            min_size=min_size,
            do_3D=do_3D,
            stitch_threshold=stitch_threshold,
        )
        if not _V4:
            eval_kwargs.update(channels=channels, channel_axis=channel_axis)

        masks, flows_orig, _ = CP.eval(image, **eval_kwargs)
        del CP
        if not do_3D and stitch_threshold == 0 and masks.ndim > 2:
            flows = [[flows_orig[0][i],
                      flows_orig[1][:, i],
                      flows_orig[2][i],
                      flows_orig[3][:, i]] for i in range(masks.shape[0])]
            masks = list(masks)
            flows_orig = flows
        return masks, flows_orig

    if not _V4:
        @thread_worker
        def compute_diameter(image, channels, model_type):
            from cellpose import models
            model_type0 = model_type if model_type in _CP_models[:4] else "cyto3"
            CP = models.Cellpose(model_type=model_type0, gpu=True)
            diam = CP.sz.eval(image, channels=channels, channel_axis=-1)[0]
            diam = np.around(diam, 2)
            del CP
            return diam

    @thread_worker
    def compute_masks(masks_orig, flows_orig, cellprob_threshold, flow_threshold):
        from cellpose.dynamics import resize_and_compute_masks

        flow_threshold = (31.0 - flow_threshold) / 10.
        if flow_threshold == 0.0:
            logger.debug('flow_threshold=0 => no masks thrown out due to model mismatch')
        logger.debug(f'computing masks with cellprob_threshold={cellprob_threshold}, flow_threshold={flow_threshold}')
        kwargs = dict(cellprob_threshold=cellprob_threshold,
                      flow_threshold=flow_threshold,
                      device="gpu")
        if not _V4:
            kwargs['resize'] = (masks_orig.shape[-2], masks_orig.shape[-1])
        result = resize_and_compute_masks(flows_orig[1], cellprob=flows_orig[2], **kwargs)
        return result[0] if not _V4 else result

    _shared_kwargs = dict(
        call_button='run segmentation',
        layout='vertical',
        diameter=dict(widget_type='LineEdit', label='diameter', value=30, tooltip='approximate diameter of cells to be segmented'),
        compute_diameter_shape=dict(widget_type='PushButton', text='compute diameter from shape layer', tooltip='create shape layer with circles and/or squares, select above, and diameter will be estimated from it'),
        cellprob_threshold=dict(widget_type='FloatSlider', name='cellprob_threshold', value=0.0, min=-8.0, max=8.0, step=0.2, tooltip='cell probability threshold (set lower to get more cells and larger cells)'),
        flow_threshold=dict(widget_type='FloatSlider', name='flow_threshold', value=0.4, min=0.0, max=3.0, step=0.05, tooltip='threshold on gradient match to accept a mask (set higher to get more cells, or to zero to turn off)'),
        min_size=dict(widget_type='SpinBox', label='min size (px)', value=15, min=1, tooltip='minimum area per mask; smaller objects are removed'),
        compute_masks_button=dict(widget_type='PushButton', text='recompute last masks with new cellprob' + ('' if _V4 else ' + model match'), enabled=False),
        resample_dynamics=dict(widget_type='CheckBox', text='resample dynamics', value=False, tooltip='if False, mask estimation with dynamics run on resized image with diameter=30; if True, flows are resized to original image size before dynamics and mask estimation (turn on for more smooth masks)'),
        process_3D=dict(widget_type='CheckBox', text='process stack as 3D', value=False, tooltip='use default 3D processing where flows in X, Y, and Z are computed and dynamics run in 3D to create masks'),
        stitch_threshold_3D=dict(widget_type='LineEdit', label='stitch threshold slices', value=0, tooltip='across time or Z, stitch together masks with IoU threshold of "stitch threshold" to create 3D segmentation'),
        clear_previous_segmentations=dict(widget_type='CheckBox', text='clear previous results', value=True),
        output_flows=dict(widget_type='CheckBox', text='output flows and cellprob', value=True),
        output_outlines=dict(widget_type='CheckBox', text='output outlines', value=True),
    )

    if not _V4:
        _mgui_kwargs = dict(
            **_shared_kwargs,
            model_type=dict(widget_type='ComboBox', label='model type', choices=[*_CP_models, 'custom'], value='cyto3', tooltip='there is a <em>cyto</em> model, a new <em>cyto2</em> model from user submissions, and a <em>nuclei</em> model'),
            custom_model=dict(widget_type='FileEdit', label='custom model path: ', tooltip='if model type is custom, specify file path to it here'),
            main_channel=dict(widget_type='ComboBox', label='channel to segment', choices=_main_channel_choices, value=0, tooltip='choose channel with cells'),
            optional_nuclear_channel=dict(widget_type='ComboBox', label='optional nuclear channel', choices=_nuclear_channel_choices, value=0, tooltip='optional, if available, choose channel with nuclei of cells'),
            compute_diameter_button=dict(widget_type='PushButton', text='compute diameter from image', tooltip='cellpose model will estimate diameter from image using specified channels'),
        )
    else:
        _mgui_kwargs = dict(
            **_shared_kwargs,
            model_type=dict(widget_type='ComboBox', visible=False, choices=[''], value='', label='model type'),
            custom_model=dict(widget_type='FileEdit', visible=False, label='custom model path'),
            main_channel=dict(widget_type='ComboBox', visible=False, choices=[0], value=0, label='channel to segment'),
            optional_nuclear_channel=dict(widget_type='ComboBox', visible=False, choices=[0], value=0, label='optional nuclear channel'),
            compute_diameter_button=dict(widget_type='PushButton', visible=False, text='compute diameter from image'),
        )

    def widget(
        viewer: Viewer,
        image_layer: Image,
        model_type,
        custom_model,
        main_channel,
        optional_nuclear_channel,
        diameter,
        shape_layer: Shapes,
        compute_diameter_shape,
        compute_diameter_button,
        cellprob_threshold,
        flow_threshold,
        min_size,
        compute_masks_button,
        resample_dynamics,
        process_3D,
        stitch_threshold_3D,
        clear_previous_segmentations,
        output_flows,
        output_outlines,
    ) -> None:

        if not hasattr(widget, 'cellpose_layers'):
            widget.cellpose_layers = []

        if clear_previous_segmentations:
            layer_names = [layer.name for layer in viewer.layers]
            for layer_name in layer_names:
                if any([cp_string in layer_name for cp_string in cp_strings]):
                    viewer.layers.remove(viewer.layers[layer_name])
            widget.cellpose_layers = []

        def _new_layers(masks, flows_orig):
            from cellpose.utils import masks_to_outlines
            from cellpose.transforms import resize_image
            import cv2

            flows = resize_image(flows_orig[0], masks.shape[-2], masks.shape[-1],
                                 interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            cellprob = resize_image(flows_orig[2], masks.shape[-2], masks.shape[-1],
                                    no_channels=True)
            cellprob = cellprob.squeeze()
            outlines = masks_to_outlines(masks) * masks
            if masks.ndim == 3 and widget.n_channels > 0:
                masks = np.repeat(np.expand_dims(masks, axis=widget.channel_axis),
                                  widget.n_channels, axis=widget.channel_axis)
                outlines = np.repeat(np.expand_dims(outlines, axis=widget.channel_axis),
                                     widget.n_channels, axis=widget.channel_axis)
                flows = np.repeat(np.expand_dims(flows, axis=widget.channel_axis),
                                  widget.n_channels, axis=widget.channel_axis)
                cellprob = np.repeat(np.expand_dims(cellprob, axis=widget.channel_axis),
                                     widget.n_channels, axis=widget.channel_axis)

            widget.flows_orig = flows_orig
            widget.masks_orig = masks
            widget.iseg = '_' + '%03d' % len(widget.cellpose_layers)
            layers = []

            if len(image_layer.scale) > 3:
                physical_scale = image_layer.scale[-3:]
            else:
                physical_scale = image_layer.scale

            if widget.output_flows.value:
                layers.append(viewer.add_image(flows, name=image_layer.name + '_cp_flows' + widget.iseg, visible=False, rgb=True, scale=physical_scale))
                layers.append(viewer.add_image(cellprob, name=image_layer.name + '_cp_cellprob' + widget.iseg, visible=False, scale=physical_scale))
            if widget.output_outlines.value:
                layers.append(viewer.add_labels(outlines, name=image_layer.name + '_cp_outlines' + widget.iseg, visible=False, scale=physical_scale))
            layers.append(viewer.add_labels(masks, name=image_layer.name + '_cp_masks' + widget.iseg, visible=False, scale=physical_scale))
            widget.cellpose_layers.append(layers)

        def _new_segmentation(segmentation):
            masks, flows_orig = segmentation
            try:
                if image_layer.ndim > 2 and not process_3D and not float(stitch_threshold_3D):
                    for mask, flow_orig in zip(masks, flows_orig):
                        _new_layers(mask, flow_orig)
                else:
                    _new_layers(masks, flows_orig)

                for layer in viewer.layers:
                    layer.visible = False
                viewer.layers[-1].visible = True
                image_layer.visible = True
                if not float(stitch_threshold_3D):
                    widget.compute_masks_button.enabled = True
            except Exception as e:
                logger.error(e)
            widget.call_button.enabled = True

        image = image_layer.data
        widget.n_channels = 0
        widget.channel_axis = None
        if image_layer.ndim == 4 and not image_layer.rgb:
            chan = np.nonzero([a == 'c' for a in viewer.dims.axis_labels])[0]
            if len(chan) > 0:
                chan = chan[0]
                widget.channel_axis = chan
                widget.n_channels = image.shape[chan]
        elif image_layer.ndim == 3 and not image_layer.rgb:
            image = image[:, :, :, np.newaxis]
        elif image_layer.rgb:
            widget.channel_axis = -1

        run_kwargs = dict(
            image=image,
            diameter=float(diameter),
            resample=resample_dynamics,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            min_size=min_size,
            do_3D=(process_3D and float(stitch_threshold_3D) == 0 and image_layer.ndim > 2),
            stitch_threshold=float(stitch_threshold_3D) if image_layer.ndim > 2 else 0.0,
        )
        if not _V4:
            run_kwargs.update(
                model_type=model_type,
                custom_model=str(custom_model.resolve()),
                channels=[max(0, main_channel), max(0, optional_nuclear_channel)],
                channel_axis=widget.channel_axis,
            )
        cp_worker = run_cellpose(**run_kwargs)
        cp_worker.returned.connect(_new_segmentation)
        cp_worker.start()

    widget = magicgui(widget, **_mgui_kwargs)

    def update_masks(masks):
        from cellpose.utils import masks_to_outlines

        outlines = masks_to_outlines(masks) * masks
        if masks.ndim == 3 and widget.n_channels > 0:
            masks = np.repeat(np.expand_dims(masks, axis=widget.channel_axis),
                              widget.n_channels, axis=widget.channel_axis)
            outlines = np.repeat(np.expand_dims(outlines, axis=widget.channel_axis),
                                 widget.n_channels, axis=widget.channel_axis)

        widget.viewer.value.layers[widget.image_layer.value.name + '_cp_masks' + widget.iseg].data = masks
        outline_str = widget.image_layer.value.name + '_cp_outlines' + widget.iseg
        if outline_str in widget.viewer.value.layers:
            widget.viewer.value.layers[outline_str].data = outlines
        widget.masks_orig = masks
        logger.debug('masks updated')

    @widget.image_layer.changed.connect
    def check_dims(image_layer):
        if image_layer.ndim == 4 and not image_layer.rgb:
            widget.process_3D.value = True
        elif image_layer.ndim == 3 and not image_layer.rgb:
            widget.process_3D.value = True
        else:
            widget.process_3D.value = False

    @widget.compute_masks_button.changed.connect
    def _compute_masks(e: Any):
        mask_worker = compute_masks(widget.masks_orig,
                                    widget.flows_orig,
                                    widget.cellprob_threshold.value,
                                    widget.flow_threshold.value)
        mask_worker.returned.connect(update_masks)
        mask_worker.start()

    def _report_diameter(diam):
        widget.diameter.value = diam
        logger.debug(f'computed diameter = {diam}')

    @widget.compute_diameter_shape.changed.connect
    def _compute_diameter_shape(e: Any):
        if widget.shape_layer.value is None:
            logger.error('no shape layer selected')
            return
        diam = 0
        k = 0
        for d in widget.shape_layer.value.data:
            if len(d) == 4:
                diam += np.ptp(d, axis=0)[-2:].sum()
                k += 2
        diam /= k
        diam *= (27 / 30)
        if k > 0:
            _report_diameter(diam)
        else:
            logger.error('no square or circle shapes created')

    if not _V4:
        @widget.compute_diameter_button.changed.connect
        def _compute_diameter(e: Any):
            if widget.model_type.value == 'custom':
                logger.error('cannot compute diameter for custom model')
            else:
                model_type = widget.model_type.value
                channels = [max(0, widget.main_channel.value), max(0, widget.optional_nuclear_channel.value)]
                image = widget.image_layer.value.data
                diam_worker = compute_diameter(image, channels, model_type)
                diam_worker.returned.connect(_report_diameter)
                diam_worker.start()

    return widget
