var extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
  hasProp = {}.hasOwnProperty;

import * as $ from "jquery";

import "bootstrap/dropdown";

import {
  button,
  span,
  ul,
  li,
  a
} from "core/dom";

import * as p from "core/properties";

import {
  AbstractButton,
  AbstractButtonView
} from "./abstract_button";

export var DropdownView = (function(superClass) {
  extend(DropdownView, superClass);

  function DropdownView() {
    return DropdownView.__super__.constructor.apply(this, arguments);
  }

  DropdownView.prototype.template = function() {
    var el;
    el = button({
      type: "button",
      disabled: this.model.disabled,
      value: this.model.default_value,
      "class": ["bk-bs-btn", "bk-bs-btn-" + this.model.button_type, "bk-bs-dropdown-toggle"]
    }, this.model.label, " ", span({
      "class": "bk-bs-caret"
    }));
    el.dataset.bkBsToggle = "dropdown";
    return el;
  };

  DropdownView.prototype.render = function() {
    var i, item, itemEl, items, label, len, link, menuEl, ref, value;
    DropdownView.__super__.render.call(this);
    this.el.classList.add("bk-bs-dropdown");
    items = [];
    ref = this.model.menu;
    for (i = 0, len = ref.length; i < len; i++) {
      item = ref[i];
      if (item != null) {
        label = item[0], value = item[1];
        link = a({}, label);
        link.dataset.value = value;
        link.addEventListener("click", (function(_this) {
          return function(e) {
            return _this.set_value(event.currentTarget.dataset.value);
          };
        })(this));
        itemEl = li({}, link);
      } else {
        itemEl = li({
          "class": "bk-bs-divider"
        });
      }
      items.push(itemEl);
    }
    menuEl = ul({
      "class": "bk-bs-dropdown-menu"
    }, items);
    this.el.appendChild(menuEl);
    $(this.buttonEl).dropdown();
    return this;
  };

  DropdownView.prototype.set_value = function(value) {
    this.buttonEl.value = this.model.value = value;
    return this.change_input();
  };

  return DropdownView;

})(AbstractButtonView);

export var Dropdown = (function(superClass) {
  extend(Dropdown, superClass);

  function Dropdown() {
    return Dropdown.__super__.constructor.apply(this, arguments);
  }

  Dropdown.prototype.type = "Dropdown";

  Dropdown.prototype.default_view = DropdownView;

  Dropdown.define({
    value: [p.String],
    default_value: [p.String],
    menu: [p.Array, []]
  });

  Dropdown.override({
    label: "Dropdown"
  });

  return Dropdown;

})(AbstractButton);