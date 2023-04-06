"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var extend = function (child, parent) { for (var key in parent) {
    if (hasProp.call(parent, key))
        child[key] = parent[key];
} function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; }, hasProp = {}.hasOwnProperty;
var annotation_1 = require("./annotation");
var signaling_1 = require("core/signaling");
var dom_1 = require("core/dom");
var p = require("core/properties");
var types_1 = require("core/util/types");
exports.BoxAnnotationView = (function (superClass) {
    extend(BoxAnnotationView, superClass);
    function BoxAnnotationView() {
        return BoxAnnotationView.__super__.constructor.apply(this, arguments);
    }
    BoxAnnotationView.prototype.initialize = function (options) {
        BoxAnnotationView.__super__.initialize.call(this, options);
        this.plot_view.canvas_overlays.appendChild(this.el);
        this.el.classList.add("bk-shading");
        return dom_1.hide(this.el);
    };
    BoxAnnotationView.prototype.connect_signals = function () {
        BoxAnnotationView.__super__.connect_signals.call(this);
        if (this.model.render_mode === 'css') {
            this.connect(this.model.change, function () {
                return this.render();
            });
            return this.connect(this.model.data_update, function () {
                return this.render();
            });
        }
        else {
            this.connect(this.model.change, (function (_this) {
                return function () {
                    return _this.plot_view.request_render();
                };
            })(this));
            return this.connect(this.model.data_update, (function (_this) {
                return function () {
                    return _this.plot_view.request_render();
                };
            })(this));
        }
    };
    BoxAnnotationView.prototype.render = function () {
        var canvas, frame, sbottom, sleft, sright, stop, xscale, yscale;
        if (!this.model.visible && this.model.render_mode === 'css') {
            dom_1.hide(this.el);
        }
        if (!this.model.visible) {
            return;
        }
        if ((this.model.left == null) && (this.model.right == null) && (this.model.top == null) && (this.model.bottom == null)) {
            dom_1.hide(this.el);
            return null;
        }
        frame = this.plot_model.frame;
        canvas = this.plot_model.canvas;
        xscale = this.plot_view.frame.xscales[this.model.x_range_name];
        yscale = this.plot_view.frame.yscales[this.model.y_range_name];
        sleft = canvas.vx_to_sx(this._calc_dim(this.model.left, this.model.left_units, xscale, frame.h_range.start));
        sright = canvas.vx_to_sx(this._calc_dim(this.model.right, this.model.right_units, xscale, frame.h_range.end));
        sbottom = canvas.vy_to_sy(this._calc_dim(this.model.bottom, this.model.bottom_units, yscale, frame.v_range.start));
        stop = canvas.vy_to_sy(this._calc_dim(this.model.top, this.model.top_units, yscale, frame.v_range.end));
        if (this.model.render_mode === 'css') {
            return this._css_box(sleft, sright, sbottom, stop);
        }
        else {
            return this._canvas_box(sleft, sright, sbottom, stop);
        }
    };
    BoxAnnotationView.prototype._css_box = function (sleft, sright, sbottom, stop) {
        var ld, sh, sw;
        sw = Math.abs(sright - sleft);
        sh = Math.abs(sbottom - stop);
        this.el.style.left = sleft + "px";
        this.el.style.width = sw + "px";
        this.el.style.top = stop + "px";
        this.el.style.height = sh + "px";
        this.el.style.borderWidth = this.model.line_width.value + "px";
        this.el.style.borderColor = this.model.line_color.value;
        this.el.style.backgroundColor = this.model.fill_color.value;
        this.el.style.opacity = this.model.fill_alpha.value;
        ld = this.model.line_dash;
        if (types_1.isArray(ld)) {
            ld = ld.length < 2 ? "solid" : "dashed";
        }
        if (types_1.isString(ld)) {
            this.el.style.borderStyle = ld;
        }
        return dom_1.show(this.el);
    };
    BoxAnnotationView.prototype._canvas_box = function (sleft, sright, sbottom, stop) {
        var ctx;
        ctx = this.plot_view.canvas_view.ctx;
        ctx.save();
        ctx.beginPath();
        ctx.rect(sleft, stop, sright - sleft, sbottom - stop);
        this.visuals.fill.set_value(ctx);
        ctx.fill();
        this.visuals.line.set_value(ctx);
        ctx.stroke();
        return ctx.restore();
    };
    BoxAnnotationView.prototype._calc_dim = function (dim, dim_units, scale, frame_extrema) {
        var vdim;
        if (dim != null) {
            if (dim_units === 'data') {
                vdim = scale.compute(dim);
            }
            else {
                vdim = dim;
            }
        }
        else {
            vdim = frame_extrema;
        }
        return vdim;
    };
    return BoxAnnotationView;
})(annotation_1.AnnotationView);
exports.BoxAnnotation = (function (superClass) {
    extend(BoxAnnotation, superClass);
    function BoxAnnotation() {
        return BoxAnnotation.__super__.constructor.apply(this, arguments);
    }
    BoxAnnotation.prototype.default_view = exports.BoxAnnotationView;
    BoxAnnotation.prototype.type = 'BoxAnnotation';
    BoxAnnotation.mixins(['line', 'fill']);
    BoxAnnotation.define({
        render_mode: [p.RenderMode, 'canvas'],
        x_range_name: [p.String, 'default'],
        y_range_name: [p.String, 'default'],
        top: [p.Number, null],
        top_units: [p.SpatialUnits, 'data'],
        bottom: [p.Number, null],
        bottom_units: [p.SpatialUnits, 'data'],
        left: [p.Number, null],
        left_units: [p.SpatialUnits, 'data'],
        right: [p.Number, null],
        right_units: [p.SpatialUnits, 'data']
    });
    BoxAnnotation.override({
        fill_color: '#fff9ba',
        fill_alpha: 0.4,
        line_color: '#cccccc',
        line_alpha: 0.3
    });
    BoxAnnotation.prototype.initialize = function (attrs, options) {
        BoxAnnotation.__super__.initialize.call(this, attrs, options);
        return this.data_update = new signaling_1.Signal(this, "data_update");
    };
    BoxAnnotation.prototype.update = function (arg) {
        var bottom, left, right, top;
        left = arg.left, right = arg.right, top = arg.top, bottom = arg.bottom;
        this.setv({
            left: left,
            right: right,
            top: top,
            bottom: bottom
        }, {
            silent: true
        });
        return this.data_update.emit();
    };
    return BoxAnnotation;
})(annotation_1.Annotation);