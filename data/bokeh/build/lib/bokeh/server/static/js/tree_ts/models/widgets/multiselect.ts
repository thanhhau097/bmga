var bind = function(fn, me){ return function(){ return fn.apply(me, arguments); }; },
  extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
  hasProp = {}.hasOwnProperty;

import * as p from "core/properties";

import {
  InputWidget,
  InputWidgetView
} from "./input_widget";

import multiselecttemplate from "./multiselecttemplate";

export var MultiSelectView = (function(superClass) {
  extend(MultiSelectView, superClass);

  function MultiSelectView() {
    this.render_selection = bind(this.render_selection, this);
    return MultiSelectView.__super__.constructor.apply(this, arguments);
  }

  MultiSelectView.prototype.template = multiselecttemplate;

  MultiSelectView.prototype.events = {
    "change select": "change_input"
  };

  MultiSelectView.prototype.initialize = function(options) {
    MultiSelectView.__super__.initialize.call(this, options);
    this.render();
    this.connect(this.model.properties.value.change, function() {
      return this.render_selection();
    });
    this.connect(this.model.properties.options.change, function() {
      return this.render();
    });
    this.connect(this.model.properties.name.change, function() {
      return this.render();
    });
    this.connect(this.model.properties.title.change, function() {
      return this.render();
    });
    return this.connect(this.model.properties.size.change, function() {
      return this.render();
    });
  };

  MultiSelectView.prototype.render = function() {
    var html;
    MultiSelectView.__super__.render.call(this);
    this.$el.empty();
    html = this.template(this.model.attributes);
    this.$el.html(html);
    this.render_selection();
    return this;
  };

  MultiSelectView.prototype.render_selection = function() {
    var i, len, ref, values, x;
    values = {};
    ref = this.model.value;
    for (i = 0, len = ref.length; i < len; i++) {
      x = ref[i];
      values[x] = true;
    }
    this.$el.find('option').each((function(_this) {
      return function(el) {
        el = _this.$el.find(el);
        if (values[el.attr('value')]) {
          return el.attr('selected', 'selected');
        }
      };
    })(this));
    return this.$el.find('select').attr('size', this.model.size);
  };

  MultiSelectView.prototype.change_input = function() {
    var is_focused, value;
    is_focused = this.$el.find('select:focus').size();
    value = this.$el.find('select').val();
    if (value) {
      this.model.value = value;
    } else {
      this.model.value = [];
    }
    MultiSelectView.__super__.change_input.call(this);
    if (is_focused) {
      return this.$el.find('select').focus();
    }
  };

  return MultiSelectView;

})(InputWidgetView);

export var MultiSelect = (function(superClass) {
  extend(MultiSelect, superClass);

  function MultiSelect() {
    return MultiSelect.__super__.constructor.apply(this, arguments);
  }

  MultiSelect.prototype.type = "MultiSelect";

  MultiSelect.prototype.default_view = MultiSelectView;

  MultiSelect.define({
    value: [p.Array, []],
    options: [p.Array, []],
    size: [p.Number, 4]
  });

  return MultiSelect;

})(InputWidget);
