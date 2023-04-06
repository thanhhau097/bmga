var extend = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
  hasProp = {}.hasOwnProperty;

import {
  ContinuousTicker
} from "./continuous_ticker";

import * as p from "core/properties";

export var SingleIntervalTicker = (function(superClass) {
  extend(SingleIntervalTicker, superClass);

  function SingleIntervalTicker() {
    return SingleIntervalTicker.__super__.constructor.apply(this, arguments);
  }

  SingleIntervalTicker.prototype.type = 'SingleIntervalTicker';

  SingleIntervalTicker.define({
    interval: [p.Number]
  });

  SingleIntervalTicker.getters({
    min_interval: function() {
      return this.interval;
    },
    max_interval: function() {
      return this.interval;
    }
  });

  SingleIntervalTicker.prototype.get_interval = function(data_low, data_high, n_desired_ticks) {
    return this.interval;
  };

  return SingleIntervalTicker;

})(ContinuousTicker);
