var extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
  hasProp = {}.hasOwnProperty;

import * as p from "core/properties";

import {
  AbstractButton,
  AbstractButtonView
} from "./abstract_button";

export var ToggleView = (function(superClass) {
  extend(ToggleView, superClass);

  function ToggleView() {
    return ToggleView.__super__.constructor.apply(this, arguments);
  }

  ToggleView.prototype.render = function() {
    ToggleView.__super__.render.call(this);
    this.buttonEl.addEventListener("click", (function(_this) {
      return function() {
        return _this.change_input();
      };
    })(this));
    if (this.model.active) {
      this.buttonEl.classList.add("bk-bs-active");
    }
    return this;
  };

  ToggleView.prototype.change_input = function() {
    this.model.active = !this.model.active;
    return ToggleView.__super__.change_input.call(this);
  };

  return ToggleView;

})(AbstractButtonView);

export var Toggle = (function(superClass) {
  extend(Toggle, superClass);

  function Toggle() {
    return Toggle.__super__.constructor.apply(this, arguments);
  }

  Toggle.prototype.type = "Toggle";

  Toggle.prototype.default_view = ToggleView;

  Toggle.define({
    active: [p.Bool, false]
  });

  Toggle.override({
    label: "Toggle"
  });

  return Toggle;

})(AbstractButton);