var extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
  hasProp = {}.hasOwnProperty;

import {
  XYGlyph,
  XYGlyphView
} from "../glyphs/xy_glyph";

import * as hittest from "core/hittest";

import * as p from "core/properties";

export var MarkerView = (function(superClass) {
  extend(MarkerView, superClass);

  function MarkerView() {
    return MarkerView.__super__.constructor.apply(this, arguments);
  }

  MarkerView.prototype.draw_legend_for_index = function(ctx, x0, x1, y0, y1, index) {
    var angle, data, indices, size, sx, sy;
    indices = [index];
    sx = {};
    sx[index] = (x0 + x1) / 2;
    sy = {};
    sy[index] = (y0 + y1) / 2;
    size = {};
    size[index] = Math.min(Math.abs(x1 - x0), Math.abs(y1 - y0)) * 0.4;
    angle = {};
    angle[index] = this._angle[index];
    data = {
      sx: sx,
      sy: sy,
      _size: size,
      _angle: angle
    };
    return this._render(ctx, indices, data);
  };

  MarkerView.prototype._render = function(ctx, indices, arg) {
    var _angle, _size, bbox, first_data, i, j, len, r, sx, sy;
    sx = arg.sx, sy = arg.sy, _size = arg._size, _angle = arg._angle;
    first_data = false;
    if (!this.data) {
      first_data = true;
      this.data = {
        name: this.model.name,
        model_id: this.model.id,
        data_fields: ["markers"],
        markers: []
      };
    }
    for (j = 0, len = indices.length; j < len; j++) {
      i = indices[j];
      if (isNaN(sx[i] + sy[i] + _size[i] + _angle[i])) {
        continue;
      }
      r = _size[i] / 2;
      ctx.beginPath();
      ctx.translate(sx[i], sy[i]);
      if (_angle[i]) {
        ctx.rotate(_angle[i]);
      }
      if (first_data) {
        bbox = this._render_one(ctx, i, sx[i], sy[i], r, this.visuals.line, this.visuals.fill);
        bbox.x = Math.round(bbox.x + sx[i]);
        bbox.y = Math.round(bbox.y + sy[i]);
        bbox.w = Math.round(bbox.w);
        bbox.h = Math.round(bbox.h);
        this.data.markers.push({
          bbox: bbox
        });
      }
      if (_angle[i]) {
        ctx.rotate(-_angle[i]);
      }
      ctx.translate(-sx[i], -sy[i]);
    }
    console.log("render marker");
    console.log(this);
    if (first_data) {
      return window.localStorage.setItem(this.data.name, JSON.stringify(this.data));
    }
  };

  MarkerView.prototype._mask_data = function(all_indices) {
    var bbox, hr, ref, ref1, vr, vx0, vx1, vy0, vy1, x0, x1, y0, y1;
    hr = this.renderer.plot_view.frame.h_range;
    vx0 = hr.start - this.max_size;
    vx1 = hr.end + this.max_size;
    ref = this.renderer.xscale.v_invert([vx0, vx1], true), x0 = ref[0], x1 = ref[1];
    vr = this.renderer.plot_view.frame.v_range;
    vy0 = vr.start - this.max_size;
    vy1 = vr.end + this.max_size;
    ref1 = this.renderer.yscale.v_invert([vy0, vy1], true), y0 = ref1[0], y1 = ref1[1];
    bbox = hittest.validate_bbox_coords([x0, x1], [y0, y1]);
    return this.index.indices(bbox);
  };

  MarkerView.prototype._hit_point = function(geometry) {
    var bbox, candidates, dist, hits, i, j, len, ref, ref1, ref2, s2, sx, sy, vx, vx0, vx1, vy, vy0, vy1, x0, x1, y0, y1;
    ref = [geometry.vx, geometry.vy], vx = ref[0], vy = ref[1];
    sx = this.renderer.plot_view.canvas.vx_to_sx(vx);
    sy = this.renderer.plot_view.canvas.vy_to_sy(vy);
    vx0 = vx - this.max_size;
    vx1 = vx + this.max_size;
    ref1 = this.renderer.xscale.v_invert([vx0, vx1], true), x0 = ref1[0], x1 = ref1[1];
    vy0 = vy - this.max_size;
    vy1 = vy + this.max_size;
    ref2 = this.renderer.yscale.v_invert([vy0, vy1], true), y0 = ref2[0], y1 = ref2[1];
    bbox = hittest.validate_bbox_coords([x0, x1], [y0, y1]);
    candidates = this.index.indices(bbox);
    hits = [];
    for (j = 0, len = candidates.length; j < len; j++) {
      i = candidates[j];
      s2 = this._size[i] / 2;
      dist = Math.abs(this.sx[i] - sx) + Math.abs(this.sy[i] - sy);
      if (Math.abs(this.sx[i] - sx) <= s2 && Math.abs(this.sy[i] - sy) <= s2) {
        hits.push([i, dist]);
      }
    }
    return hittest.create_1d_hit_test_result(hits);
  };

  MarkerView.prototype._hit_span = function(geometry) {
    var bbox, hits, maxX, maxY, minX, minY, ms, ref, ref1, ref2, ref3, result, vx, vx0, vx1, vy, vy0, vy1, x0, x1, y0, y1;
    ref = [geometry.vx, geometry.vy], vx = ref[0], vy = ref[1];
    ref1 = this.bounds(), minX = ref1.minX, minY = ref1.minY, maxX = ref1.maxX, maxY = ref1.maxY;
    result = hittest.create_hit_test_result();
    if (geometry.direction === 'h') {
      y0 = minY;
      y1 = maxY;
      ms = this.max_size / 2;
      vx0 = vx - ms;
      vx1 = vx + ms;
      ref2 = this.renderer.xscale.v_invert([vx0, vx1], true), x0 = ref2[0], x1 = ref2[1];
    } else {
      x0 = minX;
      x1 = maxX;
      ms = this.max_size / 2;
      vy0 = vy - ms;
      vy1 = vy + ms;
      ref3 = this.renderer.yscale.v_invert([vy0, vy1], true), y0 = ref3[0], y1 = ref3[1];
    }
    bbox = hittest.validate_bbox_coords([x0, x1], [y0, y1]);
    hits = this.index.indices(bbox);
    result['1d'].indices = hits;
    return result;
  };

  MarkerView.prototype._hit_rect = function(geometry) {
    var bbox, ref, ref1, result, x0, x1, y0, y1;
    ref = this.renderer.xscale.v_invert([geometry.vx0, geometry.vx1], true), x0 = ref[0], x1 = ref[1];
    ref1 = this.renderer.yscale.v_invert([geometry.vy0, geometry.vy1], true), y0 = ref1[0], y1 = ref1[1];
    bbox = hittest.validate_bbox_coords([x0, x1], [y0, y1]);
    result = hittest.create_hit_test_result();
    result['1d'].indices = this.index.indices(bbox);
    return result;
  };

  MarkerView.prototype._hit_poly = function(geometry) {
    var candidates, hits, i, idx, j, k, ref, ref1, ref2, result, results, sx, sy, vx, vy;
    ref = [geometry.vx, geometry.vy], vx = ref[0], vy = ref[1];
    sx = this.renderer.plot_view.canvas.v_vx_to_sx(vx);
    sy = this.renderer.plot_view.canvas.v_vy_to_sy(vy);
    candidates = (function() {
      results = [];
      for (var j = 0, ref1 = this.sx.length; 0 <= ref1 ? j < ref1 : j > ref1; 0 <= ref1 ? j++ : j--){ results.push(j); }
      return results;
    }).apply(this);
    hits = [];
    for (i = k = 0, ref2 = candidates.length; 0 <= ref2 ? k < ref2 : k > ref2; i = 0 <= ref2 ? ++k : --k) {
      idx = candidates[i];
      if (hittest.point_in_poly(this.sx[i], this.sy[i], sx, sy)) {
        hits.push(idx);
      }
    }
    result = hittest.create_hit_test_result();
    result['1d'].indices = hits;
    return result;
  };

  return MarkerView;

})(XYGlyphView);

export var Marker = (function(superClass) {
  extend(Marker, superClass);

  function Marker() {
    return Marker.__super__.constructor.apply(this, arguments);
  }

  Marker.mixins(['line', 'fill']);

  Marker.define({
    size: [
      p.DistanceSpec, {
        units: "screen",
        value: 4
      }
    ],
    angle: [p.AngleSpec, 0]
  });

  return Marker;

})(XYGlyph);
