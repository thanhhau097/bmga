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
exports.AnnularWedgeView = (function (superClass) {
    extend(AnnularWedgeView, superClass);
    function AnnularWedgeView() {
        return AnnularWedgeView.__super__.constructor.apply(this, arguments);
    }
    AnnularWedgeView.prototype._map_data = function () {
        var i, j, ref, results;
        if (this.model.properties.inner_radius.units === "data") {
            this.sinner_radius = this.sdist(this.renderer.xscale, this._x, this._inner_radius);
        }
        else {
            this.sinner_radius = this._inner_radius;
        }
        if (this.model.properties.outer_radius.units === "data") {
            this.souter_radius = this.sdist(this.renderer.xscale, this._x, this._outer_radius);
        }
        else {
            this.souter_radius = this._outer_radius;
        }
        this._angle = new Float32Array(this._start_angle.length);
        results = [];
        for (i = j = 0, ref = this._start_angle.length; 0 <= ref ? j < ref : j > ref; i = 0 <= ref ? ++j : --j) {
            results.push(this._angle[i] = this._end_angle[i] - this._start_angle[i]);
        }
        return results;
    };
    AnnularWedgeView.prototype._render = function (ctx, indices, arg) {
        var _angle, _start_angle, direction, i, j, len, results, sinner_radius, souter_radius, sx, sy;
        sx = arg.sx, sy = arg.sy, _start_angle = arg._start_angle, _angle = arg._angle, sinner_radius = arg.sinner_radius, souter_radius = arg.souter_radius;
        direction = this.model.properties.direction.value();
        results = [];
        for (j = 0, len = indices.length; j < len; j++) {
            i = indices[j];
            if (isNaN(sx[i] + sy[i] + sinner_radius[i] + souter_radius[i] + _start_angle[i] + _angle[i])) {
                continue;
            }
            ctx.translate(sx[i], sy[i]);
            ctx.rotate(_start_angle[i]);
            ctx.moveTo(souter_radius[i], 0);
            ctx.beginPath();
            ctx.arc(0, 0, souter_radius[i], 0, _angle[i], direction);
            ctx.rotate(_angle[i]);
            ctx.lineTo(sinner_radius[i], 0);
            ctx.arc(0, 0, sinner_radius[i], 0, -_angle[i], !direction);
            ctx.closePath();
            ctx.rotate(-_angle[i] - _start_angle[i]);
            ctx.translate(-sx[i], -sy[i]);
            if (this.visuals.fill.doit) {
                this.visuals.fill.set_vectorize(ctx, i);
                ctx.fill();
            }
            if (this.visuals.line.doit) {
                this.visuals.line.set_vectorize(ctx, i);
                results.push(ctx.stroke());
            }
            else {
                results.push(void 0);
            }
        }
        return results;
    };
    AnnularWedgeView.prototype._hit_point = function (geometry) {
        var angle, bbox, candidates, direction, dist, hits, i, ir2, j, k, len, len1, or2, ref, ref1, ref2, ref3, ref4, sx, sx0, sx1, sy, sy0, sy1, vx, vx0, vx1, vy, vy0, vy1, x, x0, x1, y, y0, y1;
        ref = [geometry.vx, geometry.vy], vx = ref[0], vy = ref[1];
        x = this.renderer.xscale.invert(vx, true);
        y = this.renderer.yscale.invert(vy, true);
        if (this.model.properties.outer_radius.units === "data") {
            x0 = x - this.max_outer_radius;
            x1 = x + this.max_outer_radius;
            y0 = y - this.max_outer_radius;
            y1 = y + this.max_outer_radius;
        }
        else {
            vx0 = vx - this.max_outer_radius;
            vx1 = vx + this.max_outer_radius;
            ref1 = this.renderer.xscale.v_invert([vx0, vx1], true), x0 = ref1[0], x1 = ref1[1];
            vy0 = vy - this.max_outer_radius;
            vy1 = vy + this.max_outer_radius;
            ref2 = this.renderer.yscale.v_invert([vy0, vy1], true), y0 = ref2[0], y1 = ref2[1];
        }
        candidates = [];
        bbox = hittest.validate_bbox_coords([x0, x1], [y0, y1]);
        ref3 = this.index.indices(bbox);
        for (j = 0, len = ref3.length; j < len; j++) {
            i = ref3[j];
            or2 = Math.pow(this.souter_radius[i], 2);
            ir2 = Math.pow(this.sinner_radius[i], 2);
            sx0 = this.renderer.xscale.compute(x, true);
            sx1 = this.renderer.xscale.compute(this._x[i], true);
            sy0 = this.renderer.yscale.compute(y, true);
            sy1 = this.renderer.yscale.compute(this._y[i], true);
            dist = Math.pow(sx0 - sx1, 2) + Math.pow(sy0 - sy1, 2);
            if (dist <= or2 && dist >= ir2) {
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
    AnnularWedgeView.prototype.draw_legend_for_index = function (ctx, x0, x1, y0, y1, index) {
        return this._generic_area_legend(ctx, x0, x1, y0, y1, index);
    };
    AnnularWedgeView.prototype._scxy = function (i) {
        var a, r;
        r = (this.sinner_radius[i] + this.souter_radius[i]) / 2;
        a = (this._start_angle[i] + this._end_angle[i]) / 2;
        return {
            x: this.sx[i] + r * Math.cos(a),
            y: this.sy[i] + r * Math.sin(a)
        };
    };
    AnnularWedgeView.prototype.scx = function (i) {
        return this._scxy(i).x;
    };
    AnnularWedgeView.prototype.scy = function (i) {
        return this._scxy(i).y;
    };
    return AnnularWedgeView;
})(xy_glyph_1.XYGlyphView);
exports.AnnularWedge = (function (superClass) {
    extend(AnnularWedge, superClass);
    function AnnularWedge() {
        return AnnularWedge.__super__.constructor.apply(this, arguments);
    }
    AnnularWedge.prototype.default_view = exports.AnnularWedgeView;
    AnnularWedge.prototype.type = 'AnnularWedge';
    AnnularWedge.mixins(['line', 'fill']);
    AnnularWedge.define({
        direction: [p.Direction, 'anticlock'],
        inner_radius: [p.DistanceSpec],
        outer_radius: [p.DistanceSpec],
        start_angle: [p.AngleSpec],
        end_angle: [p.AngleSpec]
    });
    return AnnularWedge;
})(xy_glyph_1.XYGlyph);
