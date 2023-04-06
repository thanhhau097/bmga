"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var extend = function (child, parent) { for (var key in parent) {
    if (hasProp.call(parent, key))
        child[key] = parent[key];
} function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; }, hasProp = {}.hasOwnProperty;
var button_tool_1 = require("../button_tool");
var signaling_1 = require("core/signaling");
exports.ActionToolButtonView = (function (superClass) {
    extend(ActionToolButtonView, superClass);
    function ActionToolButtonView() {
        return ActionToolButtonView.__super__.constructor.apply(this, arguments);
    }
    ActionToolButtonView.prototype._clicked = function () {
        return this.model["do"].emit();
    };
    return ActionToolButtonView;
})(button_tool_1.ButtonToolButtonView);
exports.ActionToolView = (function (superClass) {
    extend(ActionToolView, superClass);
    function ActionToolView() {
        return ActionToolView.__super__.constructor.apply(this, arguments);
    }
    ActionToolView.prototype.initialize = function (options) {
        ActionToolView.__super__.initialize.call(this, options);
        return this.connect(this.model["do"], function () {
            return this.doit();
        });
    };
    return ActionToolView;
})(button_tool_1.ButtonToolView);
exports.ActionTool = (function (superClass) {
    extend(ActionTool, superClass);
    function ActionTool() {
        return ActionTool.__super__.constructor.apply(this, arguments);
    }
    ActionTool.prototype.initialize = function (attrs, options) {
        ActionTool.__super__.initialize.call(this, attrs, options);
        return this["do"] = new signaling_1.Signal(this, "do");
    };
    return ActionTool;
})(button_tool_1.ButtonTool);