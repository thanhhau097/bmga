"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var extend = function (child, parent) { for (var key in parent) {
    if (hasProp.call(parent, key))
        child[key] = parent[key];
} function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; }, hasProp = {}.hasOwnProperty;
var xy_glyph_1 = require("./xy_glyph");
var hittest = require("core/hittest");
var p = require("core/properties");
var math_1 = require("core/util/math");
var _ = require("lodash");
exports.WedgeView = (function (superClass) {
    extend(WedgeView, superClass);
    function WedgeView() {
        return WedgeView.__super__.constructor.apply(this, arguments);
    }
    WedgeView.prototype._map_data = function () {
        if (this.model.properties.radius.units === "data") {
            return this.sradius = this.sdist(this.renderer.xscale, this._x, this._radius);
        }
        else {
            return this.sradius = this._radius;
        }
    };
    WedgeView.prototype._render = function (ctx, indices, arg) {
        var _end_angle, _start_angle, direction, get_slice_bbox, i, j, len, span, sradius, sx, sy;
        sx = arg.sx, sy = arg.sy, sradius = arg.sradius, _start_angle = arg._start_angle, _end_angle = arg._end_angle;
        direction = this.model.properties.direction.value();
        this.data = {
            name: this.renderer.model.name,
            model_id: this.model.id,
            data_fields: ["slices"],
            slices: []
        };
        get_slice_bbox = function (centre_x, centre_y, rad, start_theta, end_theta) {
            var bbox_x, bbox_y, bounding_xs, bounding_ys, cart_x1, cart_x2, cart_y1, cart_y2, i, j, k, last_quadrant, origin_bounding_xs, origin_bounding_ys, ref, ref1, theta;
            start_theta *= -1;
            end_theta *= -1;
            if (end_theta - start_theta >= Math.PI) {
                return {
                    x: centre_x - rad,
                    y: centre_y - rad,
                    w: 2 * rad,
                    h: 2 * rad
                };
            }
            cart_x1 = rad * Math.cos(-start_theta);
            cart_y1 = rad * Math.sin(-start_theta);
            cart_x2 = rad * Math.cos(-end_theta);
            cart_y2 = rad * Math.sin(-end_theta);
            origin_bounding_xs = [0, cart_x1, cart_x2];
            origin_bounding_ys = [0, cart_y1, cart_y2];
            theta = -start_theta;
            last_quadrant = null;
            if (theta > -Math.PI / 2) {
                theta = -Math.PI / 2;
                last_quadrant = 3;
            }
            else if (theta > -Math.PI) {
                theta = -Math.PI;
                last_quadrant = 2;
            }
            else if (theta > -3 * Math.PI / 2) {
                theta = -3 * Math.PI / 2;
                last_quadrant = 1;
            }
            else {
                theta = 0;
                last_quadrant = 0;
            }
            while (theta > -end_theta) {
                switch (last_quadrant) {
                    case 3:
                        origin_bounding_ys.push(-rad);
                        break;
                    case 2:
                        origin_bounding_xs.push(-rad);
                        break;
                    case 1:
                        origin_bounding_ys.push(rad);
                        break;
                    case 0:
                        origin_bounding_xs.push(rad);
                }
                if (last_quadrant > 0) {
                    last_quadrant = (last_quadrant - 1) % 4;
                }
                else {
                    last_quadrant = 0;
                }
                theta -= Math.PI / 2;
            }
            bounding_xs = [];
            for (i = j = 0, ref = origin_bounding_xs.length; 0 <= ref ? j < ref : j > ref; i = 0 <= ref ? ++j : --j) {
                bounding_xs.push(origin_bounding_xs[i] + centre_x);
            }
            bounding_ys = [];
            for (i = k = 0, ref1 = origin_bounding_ys.length; 0 <= ref1 ? k < ref1 : k > ref1; i = 0 <= ref1 ? ++k : --k) {
                bounding_ys.push(origin_bounding_ys[i] + centre_y);
            }
            bbox_x = Math.round(_.min(bounding_xs));
            bbox_y = Math.round(_.min(bounding_ys));
            return {
                x: bbox_x,
                y: bbox_y,
                w: Math.round(_.max(bounding_xs)) - bbox_x,
                h: Math.round(_.max(bounding_ys)) - bbox_y
            };
        };
        for (j = 0, len = indices.length; j < len; j++) {
            i = indices[j];
            if (isNaN(sx[i] + sy[i] + sradius[i] + _start_angle[i] + _end_angle[i])) {
                continue;
            }
            ctx.beginPath();
            ctx.arc(sx[i], sy[i], sradius[i], _start_angle[i], _end_angle[i], direction);
            ctx.lineTo(sx[i], sy[i]);
            ctx.closePath();
            if (this.visuals.fill.doit) {
                this.visuals.fill.set_vectorize(ctx, i);
                ctx.fill();
            }
            if (this.visuals.line.doit) {
                this.visuals.line.set_vectorize(ctx, i);
                ctx.stroke();
            }
            span = -(_end_angle[i] - _start_angle[i]) * 180 / Math.PI;
            this.data.slices.push({
                bbox: get_slice_bbox(sx[i], sy[i], sradius[i], _start_angle[i], _end_angle[i]),
                span: span
            });
        }
        console.log("render wedge");
        console.log(this);
        return window.localStorage.setItem(this.data.name, JSON.stringify(this.data));
    };
    WedgeView.prototype._hit_point = function (geometry) {
        var angle, bbox, candidates, direction, dist, hits, i, j, k, len, len1, r2, ref, ref1, ref2, ref3, ref4, sx, sx0, sx1, sy, sy0, sy1, vx, vx0, vx1, vy, vy0, vy1, x, x0, x1, y, y0, y1;
        ref = [geometry.vx, geometry.vy], vx = ref[0], vy = ref[1];
        x = this.renderer.xscale.invert(vx, true);
        y = this.renderer.yscale.invert(vy, true);
        if (this.model.properties.radius.units === "data") {
            x0 = x - this.max_radius;
            x1 = x + this.max_radius;
            y0 = y - this.max_radius;
            y1 = y + this.max_radius;
        }
        else {
            vx0 = vx - this.max_radius;
            vx1 = vx + this.max_radius;
            ref1 = this.renderer.xscale.v_invert([vx0, vx1], true), x0 = ref1[0], x1 = ref1[1];
            vy0 = vy - this.max_radius;
            vy1 = vy + this.max_radius;
            ref2 = this.renderer.yscale.v_invert([vy0, vy1], true), y0 = ref2[0], y1 = ref2[1];
        }
        candidates = [];
        bbox = hittest.validate_bbox_coords([x0, x1], [y0, y1]);
        ref3 = this.index.indices(bbox);
        for (j = 0, len = ref3.length; j < len; j++) {
            i = ref3[j];
            r2 = Math.pow(this.sradius[i], 2);
            sx0 = this.renderer.xscale.compute(x, true);
            sx1 = this.renderer.xscale.compute(this._x[i], true);
            sy0 = this.renderer.yscale.compute(y, true);
            sy1 = this.renderer.yscale.compute(this._y[i], true);
            dist = Math.pow(sx0 - sx1, 2) + Math.pow(sy0 - sy1, 2);
            if (dist <= r2) {
                candidates.push([i, dist]);
            }
        }
        direction = this.model.properties.direction.value();
        hits = [];
        for (k = 0, len1 = candidates.length; k < len1; k++) {
            ref4 = candidates[k], i = ref4[0], dist = ref4[1];
            sx = this.renderer.plot_view.canvas.vx_to_sx(vx);
            sy = this.renderer.plot_view.canvas.vy_to_sy(vy);
            angle = Math.atan2(sy - this.sy[i], sx - this.sx[i]);
            if (math_1.angle_between(-angle, -this._start_angle[i], -this._end_angle[i], direction)) {
                hits.push([i, dist]);
            }
        }
        return hittest.create_1d_hit_test_result(hits);
    };
    WedgeView.prototype.draw_legend_for_index = function (ctx, x0, x1, y0, y1, index) {
        return this._generic_area_legend(ctx, x0, x1, y0, y1, index);
    };
    return WedgeView;
})(xy_glyph_1.XYGlyphView);
exports.Wedge = (function (superClass) {
    extend(Wedge, superClass);
    function Wedge() {
        return Wedge.__super__.constructor.apply(this, arguments);
    }
    Wedge.prototype.default_view = exports.WedgeView;
    Wedge.prototype.type = 'Wedge';
    Wedge.mixins(['line', 'fill']);
    Wedge.define({
        direction: [p.Direction, 'anticlock'],
        radius: [p.DistanceSpec],
        start_angle: [p.AngleSpec],
        end_angle: [p.AngleSpec]
    });
    return Wedge;
})(xy_glyph_1.XYGlyph);
