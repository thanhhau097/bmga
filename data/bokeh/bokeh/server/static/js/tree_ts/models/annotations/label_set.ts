var extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
  hasProp = {}.hasOwnProperty;

import {
  TextAnnotation,
  TextAnnotationView
} from "./text_annotation";

import {
  ColumnDataSource
} from "../sources/column_data_source";

import {
  div,
  show,
  hide
} from "core/dom";

import * as p from "core/properties";

import {
  isString,
  isArray
} from "core/util/types";

export var LabelSetView = (function(superClass) {
  extend(LabelSetView, superClass);

  function LabelSetView() {
    return LabelSetView.__super__.constructor.apply(this, arguments);
  }

  LabelSetView.prototype.initialize = function(options) {
    var i, j, ref, results;
    LabelSetView.__super__.initialize.call(this, options);
    this.set_data(this.model.source);
    if (this.model.render_mode === 'css') {
      results = [];
      for (i = j = 0, ref = this._text.length; 0 <= ref ? j < ref : j > ref; i = 0 <= ref ? ++j : --j) {
        this.title_div = div({
          "class": 'bk-annotation-child',
          style: {
            display: "none"
          }
        });
        results.push(this.el.appendChild(this.title_div));
      }
      return results;
    }
  };

  LabelSetView.prototype.connect_signals = function() {
    LabelSetView.__super__.connect_signals.call(this);
    if (this.model.render_mode === 'css') {
      this.connect(this.model.change, function() {
        this.set_data(this.model.source);
        return this.render();
      });
      this.connect(this.model.source.streaming, function() {
        this.set_data(this.model.source);
        return this.render();
      });
      this.connect(this.model.source.patching, function() {
        this.set_data(this.model.source);
        return this.render();
      });
      return this.connect(this.model.source.change, function() {
        this.set_data(this.model.source);
        return this.render();
      });
    } else {
      this.connect(this.model.change, function() {
        this.set_data(this.model.source);
        return this.plot_view.request_render();
      });
      this.connect(this.model.source.streaming, function() {
        this.set_data(this.model.source);
        return this.plot_view.request_render();
      });
      this.connect(this.model.source.patching, function() {
        this.set_data(this.model.source);
        return this.plot_view.request_render();
      });
      return this.connect(this.model.source.change, function() {
        this.set_data(this.model.source);
        return this.plot_view.request_render();
      });
    }
  };

  LabelSetView.prototype.set_data = function(source) {
    LabelSetView.__super__.set_data.call(this, source);
    return this.visuals.warm_cache(source);
  };

  LabelSetView.prototype._map_data = function() {
    var sx, sy, vx, vy, xscale, yscale;
    xscale = this.plot_view.frame.xscales[this.model.x_range_name];
    yscale = this.plot_view.frame.yscales[this.model.y_range_name];
    if (this.model.x_units === "data") {
      vx = xscale.v_compute(this._x);
    } else {
      vx = this._x.slice(0);
    }
    sx = this.canvas.v_vx_to_sx(vx);
    if (this.model.y_units === "data") {
      vy = yscale.v_compute(this._y);
    } else {
      vy = this._y.slice(0);
    }
    sy = this.canvas.v_vy_to_sy(vy);
    return [sx, sy];
  };

  LabelSetView.prototype.render = function() {
    var ctx, i, j, k, ref, ref1, ref2, sx, sy;
    if (!this.model.visible && this.model.render_mode === 'css') {
      hide(this.el);
    }
    if (!this.model.visible) {
      return;
    }
    ctx = this.plot_view.canvas_view.ctx;
    this.data = {
      name: this.model.name,
      model_id: this.model.id,
      data_fields: ["labels"],
      labels: []
    };
    ref = this._map_data(), sx = ref[0], sy = ref[1];
    if (this.model.render_mode === 'canvas') {
      for (i = j = 0, ref1 = this._text.length; 0 <= ref1 ? j < ref1 : j > ref1; i = 0 <= ref1 ? ++j : --j) {
        this._v_canvas_text(ctx, i, this._text[i], sx[i] + this._x_offset[i], sy[i] - this._y_offset[i], this._angle[i]);
      }
    } else {
      for (i = k = 0, ref2 = this._text.length; 0 <= ref2 ? k < ref2 : k > ref2; i = 0 <= ref2 ? ++k : --k) {
        this._v_css_text(ctx, i, this._text[i], sx[i] + this._x_offset[i], sy[i] - this._y_offset[i], this._angle[i]);
      }
    }
    console.log("render labelset");
    console.log(this);
    return window.localStorage.setItem(this.data.name, JSON.stringify(this.data));
  };

  LabelSetView.prototype._get_size = function() {
    var ctx, height, side, width;
    ctx = this.plot_view.canvas_view.ctx;
    this.visuals.text.set_value(ctx);
    side = this.model.panel.side;
    if (side === "above" || side === "below") {
      height = ctx.measureText(this._text[0]).ascent;
      return height;
    }
    if (side === 'left' || side === 'right') {
      width = ctx.measureText(this._text[0]).width;
      return width;
    }
  };

  LabelSetView.prototype._v_canvas_text = function(ctx, i, text, sx, sy, angle) {
    var bbox, bbox_dims;
    this.visuals.text.set_vectorize(ctx, i);
    bbox_dims = this._calculate_bounding_box_dimensions(ctx, text);
    ctx.save();
    ctx.beginPath();
    ctx.translate(sx, sy);
    ctx.rotate(angle);
    ctx.rect(bbox_dims[0], bbox_dims[1], bbox_dims[2], bbox_dims[3]);
    if (this.visuals.background_fill.doit) {
      this.visuals.background_fill.set_vectorize(ctx, i);
      ctx.fill();
    }
    if (this.visuals.border_line.doit) {
      this.visuals.border_line.set_vectorize(ctx, i);
      ctx.stroke();
    }
    if (this.visuals.text.doit) {
      this.visuals.text.set_vectorize(ctx, i);
      ctx.fillText(text, 0, 0);
    }
    ctx.restore();
    bbox = {
      x: Math.round(bbox_dims[0] + sx),
      y: Math.round(bbox_dims[1] + sy),
      w: Math.round(bbox_dims[2]),
      h: 2 * Math.round(bbox_dims[3])
    };
    return this.data.labels.push({
      text: text,
      bbox: bbox
    });
  };

  LabelSetView.prototype._v_css_text = function(ctx, i, text, sx, sy, angle) {
    var bbox_dims, el, ld, line_dash;
    el = this.el.childNodes[i];
    el.textContent = text;
    this.visuals.text.set_vectorize(ctx, i);
    bbox_dims = this._calculate_bounding_box_dimensions(ctx, text);
    ld = this.visuals.border_line.line_dash.value();
    if (isArray(ld)) {
      line_dash = ld.length < 2 ? "solid" : "dashed";
    }
    if (isString(ld)) {
      line_dash = ld;
    }
    this.visuals.border_line.set_vectorize(ctx, i);
    this.visuals.background_fill.set_vectorize(ctx, i);
    el.style.position = 'absolute';
    el.style.left = (sx + bbox_dims[0]) + "px";
    el.style.top = (sy + bbox_dims[1]) + "px";
    el.style.color = "" + (this.visuals.text.text_color.value());
    el.style.opacity = "" + (this.visuals.text.text_alpha.value());
    el.style.font = "" + (this.visuals.text.font_value());
    el.style.lineHeight = "normal";
    if (angle) {
      el.style.transform = "rotate(" + angle + "rad)";
    }
    if (this.visuals.background_fill.doit) {
      el.style.backgroundColor = "" + (this.visuals.background_fill.color_value());
    }
    if (this.visuals.border_line.doit) {
      el.style.borderStyle = "" + line_dash;
      el.style.borderWidth = (this.visuals.border_line.line_width.value()) + "px";
      el.style.borderColor = "" + (this.visuals.border_line.color_value());
    }
    return show(el);
  };

  return LabelSetView;

})(TextAnnotationView);

export var LabelSet = (function(superClass) {
  extend(LabelSet, superClass);

  function LabelSet() {
    return LabelSet.__super__.constructor.apply(this, arguments);
  }

  LabelSet.prototype.default_view = LabelSetView;

  LabelSet.prototype.type = 'Label';

  LabelSet.mixins(['text', 'line:border_', 'fill:background_']);

  LabelSet.define({
    x: [p.NumberSpec],
    y: [p.NumberSpec],
    x_units: [p.SpatialUnits, 'data'],
    y_units: [p.SpatialUnits, 'data'],
    text: [
      p.StringSpec, {
        field: "text"
      }
    ],
    angle: [p.AngleSpec, 0],
    x_offset: [
      p.NumberSpec, {
        value: 0
      }
    ],
    y_offset: [
      p.NumberSpec, {
        value: 0
      }
    ],
    source: [
      p.Instance, function() {
        return new ColumnDataSource();
      }
    ],
    x_range_name: [p.String, 'default'],
    y_range_name: [p.String, 'default'],
    render_mode: [p.RenderMode, 'canvas']
  });

  LabelSet.override({
    background_fill_color: null,
    border_line_color: null
  });

  return LabelSet;

})(TextAnnotation);