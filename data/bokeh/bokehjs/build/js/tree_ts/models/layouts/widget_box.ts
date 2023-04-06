var extend1 = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
  hasProp = {}.hasOwnProperty;

import {
  logger
} from "core/logging";

import * as p from "core/properties";

import {
  extend
} from "core/util/object";

import {
  LayoutDOM,
  LayoutDOMView
} from "../layouts/layout_dom";

export var WidgetBoxView = (function(superClass) {
  extend1(WidgetBoxView, superClass);

  function WidgetBoxView() {
    return WidgetBoxView.__super__.constructor.apply(this, arguments);
  }

  WidgetBoxView.prototype.className = "bk-widget-box";

  WidgetBoxView.prototype.connect_signals = function() {
    WidgetBoxView.__super__.connect_signals.call(this);
    return this.connect(this.model.properties.children.change, (function(_this) {
      return function() {
        return _this.rebuild_child_views();
      };
    })(this));
  };

  WidgetBoxView.prototype.render = function() {
    var css_width, height, width;
    this._render_classes();
    if (this.model.sizing_mode === 'fixed' || this.model.sizing_mode === 'scale_height') {
      width = this.get_width();
      if (this.model._width.value !== width) {
        this.solver.suggest_value(this.model._width, width);
      }
    }
    if (this.model.sizing_mode === 'fixed' || this.model.sizing_mode === 'scale_width') {
      height = this.get_height();
      if (this.model._height.value !== height) {
        this.solver.suggest_value(this.model._height, height);
      }
    }
    this.solver.update_variables();
    if (this.model.sizing_mode === 'stretch_both') {
      this.el.style.position = 'absolute';
      this.el.style.left = this.model._dom_left.value + "px";
      this.el.style.top = this.model._dom_top.value + "px";
      this.el.style.width = this.model._width.value + "px";
      return this.el.style.height = this.model._height.value + "px";
    } else {
      if (this.model._width.value - 20 > 0) {
        css_width = (this.model._width.value - 20) + "px";
      } else {
        css_width = "100%";
      }
      return this.el.style.width = css_width;
    }
  };

  WidgetBoxView.prototype.get_height = function() {
    var child_view, height, key, ref;
    height = 0;
    ref = this.child_views;
    for (key in ref) {
      if (!hasProp.call(ref, key)) continue;
      child_view = ref[key];
      height += child_view.el.scrollHeight;
    }
    return height + 20;
  };

  WidgetBoxView.prototype.get_width = function() {
    var child_view, child_width, key, ref, width;
    if (this.model.width != null) {
      return this.model.width;
    } else {
      width = this.el.scrollWidth + 20;
      ref = this.child_views;
      for (key in ref) {
        if (!hasProp.call(ref, key)) continue;
        child_view = ref[key];
        child_width = child_view.el.scrollWidth;
        if (child_width > width) {
          width = child_width;
        }
      }
      return width;
    }
  };

  return WidgetBoxView;

})(LayoutDOMView);

export var WidgetBox = (function(superClass) {
  extend1(WidgetBox, superClass);

  function WidgetBox() {
    return WidgetBox.__super__.constructor.apply(this, arguments);
  }

  WidgetBox.prototype.type = 'WidgetBox';

  WidgetBox.prototype.default_view = WidgetBoxView;

  WidgetBox.prototype.initialize = function(options) {
    WidgetBox.__super__.initialize.call(this, options);
    if (this.sizing_mode === 'fixed' && this.width === null) {
      this.width = 300;
      logger.info("WidgetBox mode is fixed, but no width specified. Using default of 300.");
    }
    if (this.sizing_mode === 'scale_height') {
      return logger.warn("sizing_mode `scale_height` is not experimental for WidgetBox. Please report your results to the bokeh dev team so we can improve.");
    }
  };

  WidgetBox.prototype.get_edit_variables = function() {
    var child, edit_variables, i, len, ref;
    edit_variables = WidgetBox.__super__.get_edit_variables.call(this);
    ref = this.get_layoutable_children();
    for (i = 0, len = ref.length; i < len; i++) {
      child = ref[i];
      edit_variables = edit_variables.concat(child.get_edit_variables());
    }
    return edit_variables;
  };

  WidgetBox.prototype.get_constraints = function() {
    var child, constraints, i, len, ref;
    constraints = WidgetBox.__super__.get_constraints.call(this);
    ref = this.get_layoutable_children();
    for (i = 0, len = ref.length; i < len; i++) {
      child = ref[i];
      constraints = constraints.concat(child.get_constraints());
    }
    return constraints;
  };

  WidgetBox.prototype.get_constrained_variables = function() {
    var vars;
    vars = extend({}, WidgetBox.__super__.get_constrained_variables.call(this), {
      on_edge_align_top: this._top,
      on_edge_align_bottom: this._height_minus_bottom,
      on_edge_align_left: this._left,
      on_edge_align_right: this._width_minus_right,
      box_cell_align_top: this._top,
      box_cell_align_bottom: this._height_minus_bottom,
      box_cell_align_left: this._left,
      box_cell_align_right: this._width_minus_right,
      box_equal_size_top: this._top,
      box_equal_size_bottom: this._height_minus_bottom
    });
    if (this.sizing_mode !== 'fixed') {
      vars.box_equal_size_left = this._left;
      vars.box_equal_size_right = this._width_minus_right;
    }
    return vars;
  };

  WidgetBox.prototype.get_layoutable_children = function() {
    return this.children;
  };

  WidgetBox.define({
    children: [p.Array, []]
  });

  return WidgetBox;

})(LayoutDOM);
