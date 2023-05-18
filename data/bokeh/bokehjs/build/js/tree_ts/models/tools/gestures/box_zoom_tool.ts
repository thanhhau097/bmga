var DEFAULT_BOX_OVERLAY,
  extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
  hasProp = {}.hasOwnProperty;

import {
  GestureTool,
  GestureToolView
} from "./gesture_tool";

import {
  BoxAnnotation
} from "../../annotations/box_annotation";

import * as p from "core/properties";

export var BoxZoomToolView = (function(superClass) {
  extend(BoxZoomToolView, superClass);

  function BoxZoomToolView() {
    return BoxZoomToolView.__super__.constructor.apply(this, arguments);
  }

  BoxZoomToolView.prototype._match_aspect = function(basepoint, curpoint, frame) {
    var a, bottom, h, hend, hstart, left, ref, ref1, right, top, va, vend, vh, vstart, vw, w, xmod, ymod;
    hend = frame.h_range.end;
    hstart = frame.h_range.start;
    vend = frame.v_range.end;
    vstart = frame.v_range.start;
    w = hend - hstart;
    h = vend - vstart;
    a = w / h;
    vw = Math.abs(basepoint[0] - curpoint[0]);
    vh = Math.abs(basepoint[1] - curpoint[1]);
    if (vh === 0) {
      va = 0;
    } else {
      va = vw / vh;
    }
    if (va >= a) {
      ref = [1, va / a], xmod = ref[0], ymod = ref[1];
    } else {
      ref1 = [a / va, 1], xmod = ref1[0], ymod = ref1[1];
    }
    if (basepoint[0] <= curpoint[0]) {
      left = basepoint[0];
      right = basepoint[0] + vw * xmod;
      if (right > hend) {
        right = hend;
      }
    } else {
      right = basepoint[0];
      left = basepoint[0] - vw * xmod;
      if (left < hstart) {
        left = hstart;
      }
    }
    vw = Math.abs(right - left);
    if (basepoint[1] <= curpoint[1]) {
      bottom = basepoint[1];
      top = basepoint[1] + vw / a;
      if (top > vend) {
        top = vend;
      }
    } else {
      top = basepoint[1];
      bottom = basepoint[1] - vw / a;
      if (bottom < vstart) {
        bottom = vstart;
      }
    }
    vh = Math.abs(top - bottom);
    if (basepoint[0] <= curpoint[0]) {
      right = basepoint[0] + a * vh;
    } else {
      left = basepoint[0] - a * vh;
    }
    return [[left, right], [bottom, top]];
  };

  BoxZoomToolView.prototype._pan_start = function(e) {
    var canvas;
    canvas = this.plot_view.canvas;
    this._baseboint = [canvas.sx_to_vx(e.bokeh.sx), canvas.sy_to_vy(e.bokeh.sy)];
    return null;
  };

  BoxZoomToolView.prototype._pan = function(e) {
    var canvas, curpoint, dims, frame, ref, ref1, vx, vy;
    canvas = this.plot_view.canvas;
    curpoint = [canvas.sx_to_vx(e.bokeh.sx), canvas.sy_to_vy(e.bokeh.sy)];
    frame = this.plot_model.frame;
    dims = this.model.dimensions;
    if (this.model.match_aspect && dims === 'both') {
      ref = this._match_aspect(this._baseboint, curpoint, frame), vx = ref[0], vy = ref[1];
    } else {
      ref1 = this.model._get_dim_limits(this._baseboint, curpoint, frame, dims), vx = ref1[0], vy = ref1[1];
    }
    this.model.overlay.update({
      left: vx[0],
      right: vx[1],
      top: vy[1],
      bottom: vy[0]
    });
    return null;
  };

  BoxZoomToolView.prototype._pan_end = function(e) {
    var canvas, curpoint, dims, frame, ref, ref1, vx, vy;
    canvas = this.plot_view.canvas;
    curpoint = [canvas.sx_to_vx(e.bokeh.sx), canvas.sy_to_vy(e.bokeh.sy)];
    frame = this.plot_model.frame;
    dims = this.model.dimensions;
    if (this.model.match_aspect && dims === 'both') {
      ref = this._match_aspect(this._baseboint, curpoint, frame), vx = ref[0], vy = ref[1];
    } else {
      ref1 = this.model._get_dim_limits(this._baseboint, curpoint, frame, dims), vx = ref1[0], vy = ref1[1];
    }
    this._update(vx, vy);
    this.model.overlay.update({
      left: null,
      right: null,
      top: null,
      bottom: null
    });
    this._baseboint = null;
    return null;
  };

  BoxZoomToolView.prototype._update = function(vx, vy) {
    var end, name, ref, ref1, ref2, ref3, scale, start, xrs, yrs, zoom_info;
    if (Math.abs(vx[1] - vx[0]) <= 5 || Math.abs(vy[1] - vy[0]) <= 5) {
      return;
    }
    xrs = {};
    ref = this.plot_view.frame.xscales;
    for (name in ref) {
      scale = ref[name];
      ref1 = scale.v_invert(vx, true), start = ref1[0], end = ref1[1];
      xrs[name] = {
        start: start,
        end: end
      };
    }
    yrs = {};
    ref2 = this.plot_view.frame.yscales;
    for (name in ref2) {
      scale = ref2[name];
      ref3 = scale.v_invert(vy, true), start = ref3[0], end = ref3[1];
      yrs[name] = {
        start: start,
        end: end
      };
    }
    zoom_info = {
      xrs: xrs,
      yrs: yrs
    };
    this.plot_view.push_state('box_zoom', {
      range: zoom_info
    });
    return this.plot_view.update_range(zoom_info);
  };

  return BoxZoomToolView;

})(GestureToolView);

DEFAULT_BOX_OVERLAY = function() {
  return new BoxAnnotation({
    level: "overlay",
    render_mode: "css",
    top_units: "screen",
    left_units: "screen",
    bottom_units: "screen",
    right_units: "screen",
    fill_color: {
      value: "lightgrey"
    },
    fill_alpha: {
      value: 0.5
    },
    line_color: {
      value: "black"
    },
    line_alpha: {
      value: 1.0
    },
    line_width: {
      value: 2
    },
    line_dash: {
      value: [4, 4]
    }
  });
};

export var BoxZoomTool = (function(superClass) {
  extend(BoxZoomTool, superClass);

  function BoxZoomTool() {
    return BoxZoomTool.__super__.constructor.apply(this, arguments);
  }

  BoxZoomTool.prototype.default_view = BoxZoomToolView;

  BoxZoomTool.prototype.type = "BoxZoomTool";

  BoxZoomTool.prototype.tool_name = "Box Zoom";

  BoxZoomTool.prototype.icon = "bk-tool-icon-box-zoom";

  BoxZoomTool.prototype.event_type = "pan";

  BoxZoomTool.prototype.default_order = 20;

  BoxZoomTool.getters({
    tooltip: function() {
      return this._get_dim_tooltip(this.tool_name, this.dimensions);
    }
  });

  BoxZoomTool.define({
    dimensions: [p.Dimensions, "both"],
    overlay: [p.Instance, DEFAULT_BOX_OVERLAY],
    match_aspect: [p.Bool, false]
  });

  return BoxZoomTool;

})(GestureTool);