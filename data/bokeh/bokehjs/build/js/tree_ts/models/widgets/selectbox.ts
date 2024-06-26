var extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
  hasProp = {}.hasOwnProperty;

import {
  logger
} from "core/logging";

import * as p from "core/properties";

import {
  InputWidget,
  InputWidgetView
} from "./input_widget";

import template from "./selecttemplate";

export var SelectView = (function(superClass) {
  extend(SelectView, superClass);

  function SelectView() {
    return SelectView.__super__.constructor.apply(this, arguments);
  }

  SelectView.prototype.template = template;

  SelectView.prototype.events = {
    "change select": "change_input"
  };

  SelectView.prototype.initialize = function(options) {
    SelectView.__super__.initialize.call(this, options);
    this.render();
    return this.connect(this.model.change, function() {
      return this.render();
    });
  };

  SelectView.prototype.render = function() {
    var html;
    SelectView.__super__.render.call(this);
    this.$el.empty();
    html = this.template(this.model.attributes);
    this.$el.html(html);
    return this;
  };

  SelectView.prototype.change_input = function() {
    var value;
    value = this.$el.find('select').val();
    logger.debug("selectbox: value = " + value);
    this.model.value = value;
    return SelectView.__super__.change_input.call(this);
  };

  return SelectView;

})(InputWidgetView);

export var Select = (function(superClass) {
  extend(Select, superClass);

  function Select() {
    return Select.__super__.constructor.apply(this, arguments);
  }

  Select.prototype.type = "Select";

  Select.prototype.default_view = SelectView;

  Select.define({
    value: [p.String, ''],
    options: [p.Any, []]
  });

  return Select;

})(InputWidget);
