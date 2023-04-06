var extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
  hasProp = {}.hasOwnProperty;

import {
  Box,
  BoxView
} from "./box";

export var RowView = (function(superClass) {
  extend(RowView, superClass);

  function RowView() {
    return RowView.__super__.constructor.apply(this, arguments);
  }

  RowView.prototype.className = "bk-grid-row";

  return RowView;

})(BoxView);

export var Row = (function(superClass) {
  extend(Row, superClass);

  Row.prototype.type = 'Row';

  Row.prototype.default_view = RowView;

  function Row(attrs, options) {
    Row.__super__.constructor.call(this, attrs, options);
    this._horizontal = true;
  }

  return Row;

})(Box);
