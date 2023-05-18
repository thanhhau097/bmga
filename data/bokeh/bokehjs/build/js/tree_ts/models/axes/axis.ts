var extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
  hasProp = {}.hasOwnProperty;

import {
  SidePanel
} from "core/layout/side_panel";

import {
  GuideRenderer
} from "../renderers/guide_renderer";

import {
  RendererView
} from "../renderers/renderer";

import {
  logger
} from "core/logging";

import * as p from "core/properties";

import {
  isString,
  isArray
} from "core/util/types";

import {
  get_text_height
} from "core/util/text";

import * as _ from "lodash";

export var AxisView = (function(superClass) {
  extend(AxisView, superClass);

  function AxisView() {
    return AxisView.__super__.constructor.apply(this, arguments);
  }

  AxisView.prototype.initialize = function(options) {
    AxisView.__super__.initialize.call(this, options);
    this._x_range_name = this.model.x_range_name;
    this._y_range_name = this.model.y_range_name;
    return this.data = {
      name: this.model.attributes.name,
      model_id: this.model.id,
      model_type: "axis",
      data_fields: ["rule", "major_ticks", "minor_ticks", "major_labels", "label"],
      rule: null,
      minor_ticks: [],
      major_ticks: [],
      major_labels: [],
      label: null
    };
  };

  AxisView.prototype.render = function() {
    var ctx;
    if (this.model.visible === false) {
      return;
    }
    ctx = this.plot_view.canvas_view.ctx;
    ctx.save();
    this._draw_rule(ctx);
    this._draw_major_ticks(ctx);
    this._draw_minor_ticks(ctx);
    this._draw_major_labels(ctx);
    this._draw_axis_label(ctx);
    ctx.restore();
    console.log("render axis");
    console.log(this);
    return window.localStorage.setItem(this.data.name, JSON.stringify(this.data));
  };

  AxisView.prototype.connect_signals = function() {
    AxisView.__super__.connect_signals.call(this);
    return this.connect(this.model.change, (function(_this) {
      return function() {
        return _this.plot_view.request_render();
      };
    })(this));
  };

  AxisView.prototype._calculate_text_dimensions = function(ctx, text, font_size) {
    var height, width;
    width = ctx.measureText(text).width;
    height = get_text_height(font_size).height;
    return [width, height];
  };

  AxisView.prototype._get_size = function() {
    return this._tick_extent() + this._tick_label_extent() + this._axis_label_extent();
  };

  AxisView.prototype._draw_rule = function(ctx) {
    var bbox, bboxOff, coords, i, k, nx, ny, origX, origY, ref, ref1, ref2, ref3, ref4, ruleX, ruleY, sx, sy, x, xoff, y, yoff;
    if (!this.visuals.axis_line.doit) {
      return;
    }
    ref = coords = this.model.rule_coords, x = ref[0], y = ref[1];
    ref1 = this.plot_view.map_to_screen(x, y, this._x_range_name, this._y_range_name), sx = ref1[0], sy = ref1[1];
    ref2 = this.model.normals, nx = ref2[0], ny = ref2[1];
    ref3 = this.model.offsets, xoff = ref3[0], yoff = ref3[1];
    this.visuals.axis_line.set_value(ctx);
    ctx.beginPath();
    origX = Math.round(sx[0] + nx * xoff);
    origY = Math.round(sy[0] + ny * yoff);
    ctx.moveTo(origX, origY);
    bbox = {};
    bboxOff = 2;
    for (i = k = 1, ref4 = sx.length; 1 <= ref4 ? k < ref4 : k > ref4; i = 1 <= ref4 ? ++k : --k) {
      ruleX = Math.round(sx[i] + nx * xoff);
      ruleY = Math.round(sy[i] + ny * yoff);
      if (i === 1) {
        if (ruleX === origX) {
          bbox.x = ruleX - bboxOff;
          bbox.w = 2 * bboxOff;
        } else {
          bbox.y = ruleY - bboxOff;
          bbox.h = 2 * bboxOff;
        }
      }
      if (i === (sx.length - 1)) {
        if (_.isNil(bbox.h)) {
          bbox.y = _.min([origY, ruleY]);
          bbox.h = _.max([origY, ruleY]) - bbox.y;
        } else {
          bbox.x = _.min([origX, ruleX]);
          bbox.w = _.max([origX, ruleX]) - bbox.x;
        }
      }
      ctx.lineTo(Math.round(sx[i] + nx * xoff), Math.round(sy[i] + ny * yoff));
    }
    ctx.stroke();
    return this.data.rule = [
      {
        bbox: bbox
      }
    ];
  };

  AxisView.prototype._draw_major_ticks = function(ctx) {
    var coords, i, k, nx, ny, ref, ref1, ref2, ref3, ref4, results, sx, sy, tickH, tickW, tickX, tickY, tickYOffset, tin, tout, values, x, xoff, y, yoff;
    if (!this.visuals.major_tick_line.doit) {
      return;
    }
    coords = this.model.tick_coords;
    ref = coords.major, x = ref[0], y = ref[1];
    ref1 = this.plot_view.map_to_screen(x, y, this._x_range_name, this._y_range_name), sx = ref1[0], sy = ref1[1];
    ref2 = this.model.normals, nx = ref2[0], ny = ref2[1];
    ref3 = this.model.offsets, xoff = ref3[0], yoff = ref3[1];
    values = null;
    if (_.uniq(x).length <= 1) {
      values = y;
    } else {
      values = x;
    }
    tin = this.model.major_tick_in;
    tout = this.model.major_tick_out;
    this.visuals.major_tick_line.set_value(ctx);
    results = [];
    for (i = k = 0, ref4 = sx.length; 0 <= ref4 ? k < ref4 : k > ref4; i = 0 <= ref4 ? ++k : --k) {
      ctx.beginPath();
      ctx.moveTo(Math.round(sx[i] + nx * tout + nx * xoff), Math.round(sy[i] + ny * tout + ny * yoff));
      ctx.lineTo(Math.round(sx[i] - nx * tin + nx * xoff), Math.round(sy[i] - ny * tin + ny * yoff));
      ctx.stroke();
      tickX = Math.round(sx[i] + nx * tout + nx * xoff);
      tickY = Math.round(sy[i] + ny * tout + ny * yoff);
      tickW = Math.abs(Math.round(sx[i] - nx * tin + nx * xoff) - tickX);
      tickH = Math.abs(Math.round(sy[i] - ny * tin + ny * yoff) - tickY);
      tickYOffset = 0;
      if (tickW !== 0) {
        tickY -= tickW / 2;
        tickH = tickW;
      } else {
        tickX -= tickH / 2;
        tickW = tickH;
        tickYOffset = tickH;
      }
      results.push(this.data.major_ticks.push({
        value: values[i],
        bbox: {
          x: tickX,
          y: tickY - tickYOffset,
          w: tickW,
          h: tickH
        }
      }));
    }
    return results;
  };

  AxisView.prototype._draw_minor_ticks = function(ctx) {
    var coords, i, k, nx, ny, ref, ref1, ref2, ref3, ref4, results, sx, sy, tickH, tickW, tickX, tickY, tickYOffset, tin, tout, values, x, xoff, y, yoff;
    if (!this.visuals.minor_tick_line.doit) {
      return;
    }
    coords = this.model.tick_coords;
    ref = coords.minor, x = ref[0], y = ref[1];
    ref1 = this.plot_view.map_to_screen(x, y, this._x_range_name, this._y_range_name), sx = ref1[0], sy = ref1[1];
    ref2 = this.model.normals, nx = ref2[0], ny = ref2[1];
    ref3 = this.model.offsets, xoff = ref3[0], yoff = ref3[1];
    values = null;
    if (_.uniq(x).length <= 1) {
      values = y;
    } else {
      values = x;
    }
    tin = this.model.minor_tick_in;
    tout = this.model.minor_tick_out;
    this.visuals.minor_tick_line.set_value(ctx);
    results = [];
    for (i = k = 0, ref4 = sx.length; 0 <= ref4 ? k < ref4 : k > ref4; i = 0 <= ref4 ? ++k : --k) {
      ctx.beginPath();
      ctx.moveTo(Math.round(sx[i] + nx * tout + nx * xoff), Math.round(sy[i] + ny * tout + ny * yoff));
      ctx.lineTo(Math.round(sx[i] - nx * tin + nx * xoff), Math.round(sy[i] - ny * tin + ny * yoff));
      ctx.stroke();
      tickX = Math.round(sx[i] + nx * tout + nx * xoff);
      tickY = Math.round(sy[i] + ny * tout + ny * yoff);
      tickW = Math.abs(Math.round(sx[i] - nx * tin + nx * xoff) - tickX);
      tickH = Math.abs(Math.round(sy[i] - ny * tin + ny * yoff) - tickY);
      tickYOffset = 0;
      if (tickW !== 0) {
        tickY -= tickW / 2;
        tickH = tickW;
      } else {
        tickX -= tickH / 2;
        tickW = tickH;
        tickYOffset = tickH;
      }
      results.push(this.data.minor_ticks.push({
        value: values[i],
        bbox: {
          x: tickX,
          y: tickY - tickYOffset,
          w: tickW,
          h: tickH
        }
      }));
    }
    return results;
  };

  AxisView.prototype._draw_major_labels = function(ctx) {
    var angle, bbox, coords, dim, h, i, k, labelX, labelY, labels, nx, ny, orient, ref, ref1, ref2, ref3, ref4, ref5, results, side, standoff, sx, sy, temp, w, x, xoff, y, yoff;
    coords = this.model.tick_coords;
    ref = coords.major, x = ref[0], y = ref[1];
    ref1 = this.plot_view.map_to_screen(x, y, this._x_range_name, this._y_range_name), sx = ref1[0], sy = ref1[1];
    ref2 = this.model.normals, nx = ref2[0], ny = ref2[1];
    ref3 = this.model.offsets, xoff = ref3[0], yoff = ref3[1];
    dim = this.model.dimension;
    side = this.model.panel_side;
    orient = this.model.major_label_orientation;
    if (isString(orient)) {
      angle = this.model.panel.get_label_angle_heuristic(orient);
    } else {
      angle = -orient;
    }
    standoff = this._tick_extent() + this.model.major_label_standoff;
    labels = this.model.compute_labels(coords.major[dim]);
    this.visuals.major_label_text.set_value(ctx);
    this.model.panel.apply_label_text_heuristics(ctx, orient);
    results = [];
    for (i = k = 0, ref4 = sx.length; 0 <= ref4 ? k < ref4 : k > ref4; i = 0 <= ref4 ? ++k : --k) {
      bbox = null;
      labelX = Math.round(sx[i] + nx * standoff + nx * xoff);
      labelY = Math.round(sy[i] + ny * standoff + ny * yoff);
      if (angle) {
        ctx.translate(sx[i] + nx * standoff + nx * xoff, sy[i] + ny * standoff + ny * yoff);
        ctx.rotate(angle);
        ctx.fillText(labels[i], 0, 0);
        ctx.rotate(-angle);
        ctx.translate(-sx[i] - nx * standoff + nx * xoff, -sy[i] - ny * standoff + ny * yoff);
      } else {
        ctx.fillText(labels[i], labelX, labelY);
      }
      ref5 = this._calculate_text_dimensions(ctx, labels[i], this.visuals.major_label_text.font_value()), w = ref5[0], h = ref5[1];
      if (orient === "vertical") {
        temp = w;
        w = h;
        h = temp;
      }
      if (ny === 1) {
        labelX = Math.round(labelX - (w / 2));
      } else {
        labelX = Math.round(labelX - w);
        labelY = Math.round(labelY - (h / 2));
      }
      bbox = {
        x: labelX,
        y: labelY,
        w: w,
        h: h
      };
      results.push(this.data.major_labels.push({
        text: labels[i],
        bbox: bbox
      }));
    }
    return results;
  };

  AxisView.prototype._draw_axis_label = function(ctx) {
    var angle, bbox, h, label, nx, ny, orient, ref, ref1, ref2, ref3, ref4, side, standoff, sx, sy, temp, w, x, xoff, y, yoff;
    label = this.model.axis_label;
    if (label == null) {
      return;
    }
    ref = this.model.rule_coords, x = ref[0], y = ref[1];
    ref1 = this.plot_view.map_to_screen(x, y, this._x_range_name, this._y_range_name), sx = ref1[0], sy = ref1[1];
    ref2 = this.model.normals, nx = ref2[0], ny = ref2[1];
    ref3 = this.model.offsets, xoff = ref3[0], yoff = ref3[1];
    side = this.model.panel_side;
    orient = 'parallel';
    angle = this.model.panel.get_label_angle_heuristic(orient);
    standoff = this._tick_extent() + this._tick_label_extent() + this.model.axis_label_standoff;
    sx = (sx[0] + sx[sx.length - 1]) / 2;
    sy = (sy[0] + sy[sy.length - 1]) / 2;
    this.visuals.axis_label_text.set_value(ctx);
    this.model.panel.apply_label_text_heuristics(ctx, orient);
    x = sx + nx * standoff + nx * xoff;
    y = sy + ny * standoff + ny * yoff;
    ref4 = this._calculate_text_dimensions(ctx, label, this.visuals.axis_label_text.font_value()), w = ref4[0], h = ref4[1];
    if (isNaN(x) || isNaN(y)) {
      return;
    }
    if (angle && angle !== 0) {
      ctx.translate(x, y);
      ctx.rotate(angle);
      ctx.fillText(label, 0, 0);
      ctx.rotate(-angle);
      ctx.translate(-x, -y);
      temp = w;
      w = h;
      h = temp;
    } else {
      ctx.fillText(label, x, y);
    }
    x = Math.round(x - (w / 2));
    if (nx === -1) {
      y = Math.round(y - (h / 2));
      x = Math.round(x - (w / 4));
    }
    bbox = {
      x: x,
      y: y,
      w: w,
      h: h
    };
    return this.data.label = [
      {
        text: label,
        bbox: bbox
      }
    ];
  };

  AxisView.prototype._tick_extent = function() {
    return this.model.major_tick_out;
  };

  AxisView.prototype._tick_label_extent = function() {
    var angle, c, coords, ctx, dim, extent, h, hfactor, hscale, i, k, labels, orient, ref, s, side, val, w, wfactor;
    extent = 0;
    ctx = this.plot_view.canvas_view.ctx;
    dim = this.model.dimension;
    coords = this.model.tick_coords.major;
    side = this.model.panel_side;
    orient = this.model.major_label_orientation;
    labels = this.model.compute_labels(coords[dim]);
    this.visuals.major_label_text.set_value(ctx);
    if (isString(orient)) {
      hscale = 1;
      angle = this.model.panel.get_label_angle_heuristic(orient);
    } else {
      hscale = 2;
      angle = -orient;
    }
    angle = Math.abs(angle);
    c = Math.cos(angle);
    s = Math.sin(angle);
    if (side === "above" || side === "below") {
      wfactor = s;
      hfactor = c;
    } else {
      wfactor = c;
      hfactor = s;
    }
    for (i = k = 0, ref = labels.length; 0 <= ref ? k < ref : k > ref; i = 0 <= ref ? ++k : --k) {
      if (labels[i] == null) {
        continue;
      }
      w = ctx.measureText(labels[i]).width * 1.1;
      h = ctx.measureText(labels[i]).ascent * 0.9;
      val = w * wfactor + (h / hscale) * hfactor;
      if (val > extent) {
        extent = val;
      }
    }
    if (extent > 0) {
      extent += this.model.major_label_standoff;
    }
    return extent;
  };

  AxisView.prototype._axis_label_extent = function() {
    var angle, axis_label, c, ctx, extent, h, orient, s, side, w;
    extent = 0;
    side = this.model.panel_side;
    axis_label = this.model.axis_label;
    orient = 'parallel';
    ctx = this.plot_view.canvas_view.ctx;
    this.visuals.axis_label_text.set_value(ctx);
    angle = Math.abs(this.model.panel.get_label_angle_heuristic(orient));
    c = Math.cos(angle);
    s = Math.sin(angle);
    if (axis_label) {
      extent += this.model.axis_label_standoff;
      this.visuals.axis_label_text.set_value(ctx);
      w = ctx.measureText(axis_label).width * 1.1;
      h = ctx.measureText(axis_label).ascent * 0.9;
      if (side === "above" || side === "below") {
        extent += w * s + h * c;
      } else {
        extent += w * c + h * s;
      }
    }
    return extent;
  };

  return AxisView;

})(RendererView);

export var Axis = (function(superClass) {
  extend(Axis, superClass);

  function Axis() {
    return Axis.__super__.constructor.apply(this, arguments);
  }

  Axis.prototype.default_view = AxisView;

  Axis.prototype.type = 'Axis';

  Axis.mixins(['line:axis_', 'line:major_tick_', 'line:minor_tick_', 'text:major_label_', 'text:axis_label_']);

  Axis.define({
    bounds: [p.Any, 'auto'],
    ticker: [p.Instance, null],
    formatter: [p.Instance, null],
    x_range_name: [p.String, 'default'],
    y_range_name: [p.String, 'default'],
    axis_label: [p.String, ''],
    axis_label_standoff: [p.Int, 5],
    major_label_standoff: [p.Int, 5],
    major_label_orientation: [p.Any, "horizontal"],
    major_label_overrides: [p.Any, {}],
    major_tick_in: [p.Number, 2],
    major_tick_out: [p.Number, 6],
    minor_tick_in: [p.Number, 0],
    minor_tick_out: [p.Number, 4]
  });

  Axis.override({
    axis_line_color: 'black',
    major_tick_line_color: 'black',
    minor_tick_line_color: 'black',
    major_label_text_font_size: "8pt",
    major_label_text_align: "center",
    major_label_text_baseline: "alphabetic",
    axis_label_text_font_size: "10pt",
    axis_label_text_font_style: "italic"
  });

  Axis.internal({
    panel_side: [p.Any]
  });

  Axis.prototype.compute_labels = function(ticks) {
    var i, k, labels, ref;
    labels = this.formatter.doFormat(ticks, this);
    for (i = k = 0, ref = ticks.length; 0 <= ref ? k < ref : k > ref; i = 0 <= ref ? ++k : --k) {
      if (ticks[i] in this.major_label_overrides) {
        labels[i] = this.major_label_overrides[ticks[i]];
      }
    }
    return labels;
  };

  Axis.getters({
    computed_bounds: function() {
      return this._computed_bounds();
    },
    rule_coords: function() {
      return this._rule_coords();
    },
    tick_coords: function() {
      return this._tick_coords();
    },
    ranges: function() {
      return this._ranges();
    },
    normals: function() {
      return this.panel._normals;
    },
    dimension: function() {
      return this.panel._dim;
    },
    offsets: function() {
      return this._offsets();
    },
    loc: function() {
      return this._get_loc();
    }
  });

  Axis.prototype.add_panel = function(side) {
    this.panel = new SidePanel({
      side: side
    });
    this.panel.attach_document(this.document);
    return this.panel_side = side;
  };

  Axis.prototype._offsets = function() {
    var frame, ref, side, xoff, yoff;
    side = this.panel_side;
    ref = [0, 0], xoff = ref[0], yoff = ref[1];
    frame = this.plot.plot_canvas.frame;
    switch (side) {
      case "below":
        yoff = Math.abs(this.panel._top.value - frame._bottom.value);
        break;
      case "above":
        yoff = Math.abs(this.panel._bottom.value - frame._top.value);
        break;
      case "right":
        xoff = Math.abs(this.panel._left.value - frame._right.value);
        break;
      case "left":
        xoff = Math.abs(this.panel._right.value - frame._left.value);
    }
    return [xoff, yoff];
  };

  Axis.prototype._ranges = function() {
    var frame, i, j, ranges;
    i = this.dimension;
    j = (i + 1) % 2;
    frame = this.plot.plot_canvas.frame;
    ranges = [frame.x_ranges[this.x_range_name], frame.y_ranges[this.y_range_name]];
    return [ranges[i], ranges[j]];
  };

  Axis.prototype._computed_bounds = function() {
    var cross_range, end, range, range_bounds, ref, ref1, start, user_bounds;
    ref = this.ranges, range = ref[0], cross_range = ref[1];
    user_bounds = (ref1 = this.bounds) != null ? ref1 : 'auto';
    range_bounds = [range.min, range.max];
    if (user_bounds === 'auto') {
      return range_bounds;
    }
    if (isArray(user_bounds)) {
      if (Math.abs(user_bounds[0] - user_bounds[1]) > Math.abs(range_bounds[0] - range_bounds[1])) {
        start = Math.max(Math.min(user_bounds[0], user_bounds[1]), range_bounds[0]);
        end = Math.min(Math.max(user_bounds[0], user_bounds[1]), range_bounds[1]);
      } else {
        start = Math.min(user_bounds[0], user_bounds[1]);
        end = Math.max(user_bounds[0], user_bounds[1]);
      }
      return [start, end];
    }
    logger.error("user bounds '" + user_bounds + "' not understood");
    return null;
  };

  Axis.prototype._rule_coords = function() {
    var coords, cross_range, end, i, j, range, ref, ref1, start, xs, ys;
    i = this.dimension;
    j = (i + 1) % 2;
    ref = this.ranges, range = ref[0], cross_range = ref[1];
    ref1 = this.computed_bounds, start = ref1[0], end = ref1[1];
    xs = new Array(2);
    ys = new Array(2);
    coords = [xs, ys];
    coords[i][0] = Math.max(start, range.min);
    coords[i][1] = Math.min(end, range.max);
    if (coords[i][0] > coords[i][1]) {
      coords[i][0] = coords[i][1] = 0/0;
    }
    coords[j][0] = this.loc;
    coords[j][1] = this.loc;
    return coords;
  };

  Axis.prototype._tick_coords = function() {
    var coords, cross_range, end, i, ii, j, k, l, m, majors, minor_coords, minor_xs, minor_ys, minors, range, range_max, range_min, ref, ref1, ref2, ref3, ref4, ref5, start, ticks, xs, ys;
    i = this.dimension;
    j = (i + 1) % 2;
    ref = this.ranges, range = ref[0], cross_range = ref[1];
    ref1 = this.computed_bounds, start = ref1[0], end = ref1[1];
    ticks = this.ticker.get_ticks(start, end, range, this.loc, {});
    majors = ticks.major;
    minors = ticks.minor;
    xs = [];
    ys = [];
    coords = [xs, ys];
    minor_xs = [];
    minor_ys = [];
    minor_coords = [minor_xs, minor_ys];
    if (range.type === "FactorRange") {
      for (ii = k = 0, ref2 = majors.length; 0 <= ref2 ? k < ref2 : k > ref2; ii = 0 <= ref2 ? ++k : --k) {
        coords[i].push(majors[ii]);
        coords[j].push(this.loc);
      }
    } else {
      ref3 = [range.min, range.max], range_min = ref3[0], range_max = ref3[1];
      for (ii = l = 0, ref4 = majors.length; 0 <= ref4 ? l < ref4 : l > ref4; ii = 0 <= ref4 ? ++l : --l) {
        if (majors[ii] < range_min || majors[ii] > range_max) {
          continue;
        }
        coords[i].push(majors[ii]);
        coords[j].push(this.loc);
      }
      for (ii = m = 0, ref5 = minors.length; 0 <= ref5 ? m < ref5 : m > ref5; ii = 0 <= ref5 ? ++m : --m) {
        if (minors[ii] < range_min || minors[ii] > range_max) {
          continue;
        }
        minor_coords[i].push(minors[ii]);
        minor_coords[j].push(this.loc);
      }
    }
    return {
      "major": coords,
      "minor": minor_coords
    };
  };

  Axis.prototype._get_loc = function() {
    var cend, cross_range, cstart, range, ref, side;
    ref = this.ranges, range = ref[0], cross_range = ref[1];
    cstart = cross_range.start;
    cend = cross_range.end;
    side = this.panel_side;
    switch (side) {
      case 'left':
      case 'below':
        return cross_range.start;
      case 'right':
      case 'above':
        return cross_range.end;
    }
  };

  return Axis;

})(GuideRenderer);