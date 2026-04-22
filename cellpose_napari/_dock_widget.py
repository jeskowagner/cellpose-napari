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
                     model_type=None, custom_model=None, channels=None, channel_axis=None,
                     z_axis=None):
        from cellpose import models

        if _V4:
            CP = models.CellposeModel(gpu=True)
        elif model_type == 'custom':
            CP = models.CellposeModel(pretrained_model=custom_model, gpu=True)
        else:
            CP = models.CellposeModel(model_type=model_type, gpu=True)

        if _V4:
            eval_kwargs = dict(
                diameter=diameter,
                resample=resample,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
                min_size=min_size,
                do_3D=do_3D,
                stitch_threshold=stitch_threshold,
            )
            if channel_axis is not None:
                eval_kwargs['channel_axis'] = channel_axis
            if z_axis is not None:
                eval_kwargs['z_axis'] = z_axis
        else:
            eval_kwargs = dict(
                diameter=diameter,
                resample=resample,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
                min_size=min_size,
                do_3D=do_3D,
                stitch_threshold=stitch_threshold,
                channels=channels,
                channel_axis=channel_axis,
            )

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
                      flow_threshold=flow_threshold)
        if not _V4:
            kwargs['resize'] = (masks_orig.shape[-2], masks_orig.shape[-1])
        result = resize_and_compute_masks(flows_orig[1], cellprob=flows_orig[2], **kwargs)
        return result[0] if not _V4 else result

    # -1 is a sentinel meaning "not specified" for z_axis and channel_axis.
    # This lets a single ComboBox serve both 3D (optional) and 4D (required) images.
    _z_axis_3d_choices = [('none', -1), ('0', 0), ('1', 1), ('2', 2)]
    _channel_axis_3d_choices = [('none (ZYX)', -1), ('0', 0), ('1', 1), ('2', 2)]

    _shared_kwargs = dict(
        call_button='run segmentation',
        layout='vertical',
        diameter=dict(widget_type='LineEdit', label='diameter', value=30,
                      tooltip='approximate diameter of cells to be segmented'),
        compute_diameter_shape=dict(widget_type='PushButton', text='compute diameter from shape layer',
                                    tooltip='create shape layer with circles and/or squares, select above, '
                                            'and diameter will be estimated from it'),
        cellprob_threshold=dict(widget_type='FloatSlider', name='cellprob_threshold', value=0.0,
                                min=-8.0, max=8.0, step=0.2,
                                tooltip='cell probability threshold (set lower to get more cells and larger cells)'),
        flow_threshold=dict(widget_type='FloatSlider', name='flow_threshold', value=0.4,
                            min=0.0, max=3.0, step=0.05,
                            tooltip='threshold on gradient match to accept a mask '
                                    '(set higher to get more cells, or to zero to turn off)'),
        min_size=dict(widget_type='SpinBox', label='min size (px)', value=15, min=1,
                      tooltip='minimum area per mask; smaller objects are removed'),
        compute_masks_button=dict(widget_type='PushButton',
                                  text='recompute last masks with new cellprob' + ('' if _V4 else ' + model match'),
                                  enabled=False),
        resample_dynamics=dict(widget_type='CheckBox', text='resample dynamics', value=False,
                               tooltip='if False, mask estimation with dynamics run on resized image with diameter=30; '
                                       'if True, flows are resized to original image size before dynamics and mask '
                                       'estimation (turn on for more smooth masks)'),
        process_3D=dict(widget_type='CheckBox', text='process stack as 3D', value=False,
                        tooltip='use default 3D processing where flows in X, Y, and Z are computed '
                                'and dynamics run in 3D to create masks'),
        stitch_threshold_3D=dict(widget_type='LineEdit', label='stitch threshold slices', value=0,
                                 tooltip='across time or Z, stitch together masks with IoU threshold of '
                                         '"stitch threshold" to create 3D segmentation'),
        clear_previous_segmentations=dict(widget_type='CheckBox', text='clear previous results', value=True),
    )

    if not _V4:
        _mgui_kwargs = dict(
            **_shared_kwargs,
            model_type=dict(widget_type='ComboBox', label='model type', choices=[*_CP_models, 'custom'],
                            value='cyto3',
                            tooltip='there is a <em>cyto</em> model, a new <em>cyto2</em> model from user '
                                    'submissions, and a <em>nuclei</em> model'),
            custom_model=dict(widget_type='FileEdit', label='custom model path: ',
                              tooltip='if model type is custom, specify file path to it here'),
            main_channel=dict(widget_type='ComboBox', label='channel to segment',
                              choices=_main_channel_choices, value=0, tooltip='choose channel with cells'),
            optional_nuclear_channel=dict(widget_type='ComboBox', label='optional nuclear channel',
                                          choices=_nuclear_channel_choices, value=0,
                                          tooltip='optional, if available, choose channel with nuclei of cells'),
            compute_diameter_button=dict(widget_type='PushButton', text='compute diameter from image',
                                         tooltip='cellpose model will estimate diameter from image using '
                                                 'specified channels'),
            z_axis=dict(widget_type='SpinBox', label='z axis', value=0, visible=False),
            channel_axis=dict(widget_type='SpinBox', label='channel axis', value=-1, visible=False),
        )
    else:
        # Build V4 kwargs, inserting z_axis and channel_axis immediately after process_3D
        _mgui_kwargs = {}
        for k, v in _shared_kwargs.items():
            _mgui_kwargs[k] = v
            if k == 'process_3D':
                _mgui_kwargs['z_axis'] = dict(
                    widget_type='ComboBox', label='z axis', choices=[0, 1, 2, 3], value=0,
                    visible=False, tooltip='which axis represents Z in the 4D image')
                _mgui_kwargs['channel_axis'] = dict(
                    widget_type='ComboBox', label='channel axis', choices=_channel_axis_3d_choices,
                    value=-1, visible=False,
                    tooltip='which axis represents channels; "none" treats all axes as spatial (ZYX)')
        # Hidden stubs so the shared function signature stays valid under V4
        _mgui_kwargs.update(
            model_type=dict(widget_type='ComboBox', visible=False, choices=[''], value='', label='model type'),
            custom_model=dict(widget_type='FileEdit', visible=False, label='custom model path'),
            main_channel=dict(widget_type='ComboBox', visible=False, choices=[0], value=0,
                              label='channel to segment'),
            optional_nuclear_channel=dict(widget_type='ComboBox', visible=False, choices=[0], value=0,
                                          label='optional nuclear channel'),
            compute_diameter_button=dict(widget_type='PushButton', visible=False,
                                         text='compute diameter from image'),
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
        z_axis,
        channel_axis,
        stitch_threshold_3D,
        clear_previous_segmentations,
    ) -> None:

        if not hasattr(widget, 'cellpose_layers'):
            widget.cellpose_layers = []

        if clear_previous_segmentations:
            layer_names = [layer.name for layer in viewer.layers]
            for layer_name in layer_names:
                if any(cp_string in layer_name for cp_string in cp_strings):
                    viewer.layers.remove(viewer.layers[layer_name])
            widget.cellpose_layers = []

        def _new_layers(masks, flows_orig):
            from cellpose.utils import masks_to_outlines
            import cv2

            if masks.ndim == 2:
                from cellpose.transforms import resize_image
                flows = resize_image(flows_orig[0], masks.shape[-2], masks.shape[-1],
                                     interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                cellprob = resize_image(flows_orig[2], masks.shape[-2], masks.shape[-1],
                                        no_channels=True).squeeze()
            else:
                # 3D: cellpose returns flows at the original resolution; resize_image
                # would collapse the Z axis, so use the arrays directly.
                flows = flows_orig[0].astype(np.uint8)
                cellprob = flows_orig[2]
            outlines = masks_to_outlines(masks) * masks
            if masks.ndim == 3 and widget._n_channels > 0:
                ax = widget._channel_axis
                n = widget._n_channels
                masks = np.repeat(np.expand_dims(masks, axis=ax), n, axis=ax)
                outlines = np.repeat(np.expand_dims(outlines, axis=ax), n, axis=ax)
                flows = np.repeat(np.expand_dims(flows, axis=ax), n, axis=ax)
                cellprob = np.repeat(np.expand_dims(cellprob, axis=ax), n, axis=ax)

            widget.flows_orig = flows_orig
            widget.masks_orig = masks
            widget.iseg = '_' + '%03d' % len(widget.cellpose_layers)

            physical_scale = image_layer.scale[-3:] if len(image_layer.scale) > 3 else image_layer.scale
            name = image_layer.name
            layers = [
                viewer.add_image(flows, name=name + '_cp_flows' + widget.iseg,
                                 visible=False, rgb=True, scale=physical_scale),
                viewer.add_image(cellprob, name=name + '_cp_cellprob' + widget.iseg,
                                 visible=False, scale=physical_scale),
                viewer.add_labels(outlines, name=name + '_cp_outlines' + widget.iseg,
                                  visible=False, scale=physical_scale),
                viewer.add_labels(masks, name=name + '_cp_masks' + widget.iseg,
                                  visible=False, scale=physical_scale),
            ]
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

        # -1 is the sentinel meaning "not specified" for both axis selectors
        _channel_axis = channel_axis if channel_axis >= 0 else None
        _z_axis = z_axis if z_axis >= 0 else None

        image = image_layer.data
        widget._n_channels = 0
        widget._channel_axis = None

        if image_layer.ndim == 4 and not image_layer.rgb:
            if _V4:
                widget._channel_axis = _channel_axis
                if _channel_axis is not None:
                    widget._n_channels = image.shape[_channel_axis]
            else:
                chan = np.nonzero([a == 'c' for a in viewer.dims.axis_labels])[0]
                if len(chan) > 0:
                    chan = int(chan[0])
                    widget._channel_axis = chan
                    widget._n_channels = image.shape[chan]
        elif image_layer.ndim == 3 and not image_layer.rgb:
            if _V4:
                widget._channel_axis = _channel_axis
                if _channel_axis is not None:
                    widget._n_channels = image.shape[_channel_axis]
            else:
                image = image[:, :, :, np.newaxis]
        elif image_layer.rgb:
            widget._channel_axis = -1

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
        if _V4:
            if image_layer.ndim == 4 and not image_layer.rgb:
                run_kwargs['z_axis'] = z_axis
                run_kwargs['channel_axis'] = _channel_axis
            elif image_layer.ndim == 3 and not image_layer.rgb:
                if _z_axis is not None:
                    run_kwargs['z_axis'] = _z_axis
                if _channel_axis is not None:
                    run_kwargs['channel_axis'] = _channel_axis
        else:
            run_kwargs.update(
                model_type=model_type,
                custom_model=str(custom_model.resolve()),
                channels=[max(0, main_channel), max(0, optional_nuclear_channel)],
                channel_axis=widget._channel_axis,
            )

        cp_worker = run_cellpose(**run_kwargs)
        cp_worker.returned.connect(_new_segmentation)
        cp_worker.start()

    widget = magicgui(widget, **_mgui_kwargs)

    def update_masks(masks):
        from cellpose.utils import masks_to_outlines

        outlines = masks_to_outlines(masks) * masks
        if masks.ndim == 3 and widget._n_channels > 0:
            ax = widget._channel_axis
            n = widget._n_channels
            masks = np.repeat(np.expand_dims(masks, axis=ax), n, axis=ax)
            outlines = np.repeat(np.expand_dims(outlines, axis=ax), n, axis=ax)

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
            if _V4:
                axis_labels = image_layer.axis_labels
                chan = np.nonzero([a == 'c' for a in axis_labels])[0]
                chan = int(chan[0]) if len(chan) > 0 else None
                z_val = np.nonzero([a == 'z' for a in axis_labels])[0]
                z_val = int(z_val[0]) if len(z_val) > 0 else None

                all_axes = list(range(image_layer.ndim))
                z_choices = [i for i in all_axes if i != chan] if chan is not None else all_axes
                c_choices = [i for i in all_axes if i != z_val] if z_val is not None else all_axes

                widget.z_axis.choices = z_choices
                widget.z_axis.value = z_val if z_val in z_choices else z_choices[0]
                widget.z_axis.visible = True

                widget.channel_axis.choices = c_choices
                widget.channel_axis.value = chan if chan in c_choices else c_choices[0]
                widget.channel_axis.visible = True

        elif image_layer.ndim == 3 and not image_layer.rgb:
            widget.process_3D.value = True
            if _V4:
                axis_labels = image_layer.axis_labels
                chan = np.nonzero([a == 'c' for a in axis_labels])[0]
                z_val = np.nonzero([a == 'z' for a in axis_labels])[0]
                default_chan = int(chan[0]) if len(chan) > 0 else -1
                default_z = int(z_val[0]) if len(z_val) > 0 else 0

                widget.z_axis.choices = _z_axis_3d_choices
                widget.z_axis.value = default_z
                widget.z_axis.visible = True

                widget.channel_axis.choices = _channel_axis_3d_choices
                widget.channel_axis.value = default_chan
                widget.channel_axis.visible = True

        else:
            widget.process_3D.value = False
            if _V4:
                widget.z_axis.visible = False
                widget.channel_axis.visible = False

    if widget.image_layer.value is not None:
        check_dims(widget.image_layer.value)

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
