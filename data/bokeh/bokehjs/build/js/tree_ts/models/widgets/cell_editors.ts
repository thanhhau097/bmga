var extend1 = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
  hasProp = {}.hasOwnProperty;

import * as $ from "jquery";

import "jquery-ui/autocomplete";

import "jquery-ui/spinner";

import * as p from "core/properties";

import {
  extend
} from "core/util/object";

import {
  DOMView
} from "core/dom_view";

import {
  Model
} from "../../model";

import {
  DTINDEX_NAME
} from "./data_table";

import {
  JQueryable
} from "./jqueryable";

export var CellEditorView = (function(superClass) {
  extend1(CellEditorView, superClass);

  extend1(CellEditorView.prototype, JQueryable);

  CellEditorView.prototype.className = "bk-cell-editor";

  CellEditorView.prototype.input = null;

  CellEditorView.prototype.emptyValue = null;

  CellEditorView.prototype.defaultValue = null;

  function CellEditorView(options) {
    this.args = options;
    CellEditorView.__super__.constructor.call(this, extend({
      model: options.column.editor
    }, options));
  }

  CellEditorView.prototype.initialize = function(options) {
    CellEditorView.__super__.initialize.call(this, options);
    return this.render();
  };

  CellEditorView.prototype.render = function() {
    CellEditorView.__super__.render.call(this);
    this.$el.appendTo(this.args.container);
    this.$input = $(this.input);
    this.$el.append(this.$input);
    this.renderEditor();
    this.disableNavigation();
    this._prefix_ui();
    return this;
  };

  CellEditorView.prototype.renderEditor = function() {};

  CellEditorView.prototype.disableNavigation = function() {
    return this.$input.keydown((function(_this) {
      return function(event) {
        var stop;
        stop = function() {
          return event.stopImmediatePropagation();
        };
        switch (event.keyCode) {
          case $.ui.keyCode.LEFT:
            return stop();
          case $.ui.keyCode.RIGHT:
            return stop();
          case $.ui.keyCode.UP:
            return stop();
          case $.ui.keyCode.DOWN:
            return stop();
          case $.ui.keyCode.PAGE_UP:
            return stop();
          case $.ui.keyCode.PAGE_DOWN:
            return stop();
        }
      };
    })(this));
  };

  CellEditorView.prototype.destroy = function() {
    return this.remove();
  };

  CellEditorView.prototype.focus = function() {
    return this.$input.focus();
  };

  CellEditorView.prototype.show = function() {};

  CellEditorView.prototype.hide = function() {};

  CellEditorView.prototype.position = function() {};

  CellEditorView.prototype.getValue = function() {
    return this.$input.val();
  };

  CellEditorView.prototype.setValue = function(val) {
    return this.$input.val(val);
  };

  CellEditorView.prototype.serializeValue = function() {
    return this.getValue();
  };

  CellEditorView.prototype.isValueChanged = function() {
    return !(this.getValue() === "" && (this.defaultValue == null)) && (this.getValue() !== this.defaultValue);
  };

  CellEditorView.prototype.applyValue = function(item, state) {
    return this.args.grid.getData().setField(item[DTINDEX_NAME], this.args.column.field, state);
  };

  CellEditorView.prototype.loadValue = function(item) {
    var value;
    value = item[this.args.column.field];
    this.defaultValue = value != null ? value : this.emptyValue;
    return this.setValue(this.defaultValue);
  };

  CellEditorView.prototype.validateValue = function(value) {
    var result;
    if (this.args.column.validator) {
      result = this.args.column.validator(value);
      if (!result.valid) {
        return result;
      }
    }
    return {
      valid: true,
      msg: null
    };
  };

  CellEditorView.prototype.validate = function() {
    return this.validateValue(this.getValue());
  };

  return CellEditorView;

})(DOMView);

export var CellEditor = (function(superClass) {
  extend1(CellEditor, superClass);

  function CellEditor() {
    return CellEditor.__super__.constructor.apply(this, arguments);
  }

  CellEditor.prototype.type = "CellEditor";

  CellEditor.prototype.default_view = CellEditorView;

  return CellEditor;

})(Model);

export var StringEditorView = (function(superClass) {
  extend1(StringEditorView, superClass);

  function StringEditorView() {
    return StringEditorView.__super__.constructor.apply(this, arguments);
  }

  StringEditorView.prototype.emptyValue = "";

  StringEditorView.prototype.input = '<input type="text" />';

  StringEditorView.prototype.renderEditor = function() {
    var completions;
    completions = this.model.completions;
    if (completions.length !== 0) {
      this.$input.autocomplete({
        source: completions
      });
      this.$input.autocomplete("widget").addClass("bk-cell-editor-completion");
    }
    return this.$input.focus().select();
  };

  StringEditorView.prototype.loadValue = function(item) {
    StringEditorView.__super__.loadValue.call(this, item);
    this.$input[0].defaultValue = this.defaultValue;
    return this.$input.select();
  };

  return StringEditorView;

})(CellEditorView);

export var StringEditor = (function(superClass) {
  extend1(StringEditor, superClass);

  function StringEditor() {
    return StringEditor.__super__.constructor.apply(this, arguments);
  }

  StringEditor.prototype.type = 'StringEditor';

  StringEditor.prototype.default_view = StringEditorView;

  StringEditor.define({
    completions: [p.Array, []]
  });

  return StringEditor;

})(CellEditor);

export var TextEditorView = (function(superClass) {
  extend1(TextEditorView, superClass);

  function TextEditorView() {
    return TextEditorView.__super__.constructor.apply(this, arguments);
  }

  return TextEditorView;

})(CellEditorView);

export var TextEditor = (function(superClass) {
  extend1(TextEditor, superClass);

  function TextEditor() {
    return TextEditor.__super__.constructor.apply(this, arguments);
  }

  TextEditor.prototype.type = 'TextEditor';

  TextEditor.prototype.default_view = TextEditorView;

  return TextEditor;

})(CellEditor);

export var SelectEditorView = (function(superClass) {
  extend1(SelectEditorView, superClass);

  function SelectEditorView() {
    return SelectEditorView.__super__.constructor.apply(this, arguments);
  }

  SelectEditorView.prototype.input = '<select />';

  SelectEditorView.prototype.renderEditor = function() {
    var i, len, option, ref;
    ref = this.model.options;
    for (i = 0, len = ref.length; i < len; i++) {
      option = ref[i];
      this.$input.append($('<option>').attr({
        value: option
      }).text(option));
    }
    return this.focus();
  };

  SelectEditorView.prototype.loadValue = function(item) {
    SelectEditorView.__super__.loadValue.call(this, item);
    return this.$input.select();
  };

  return SelectEditorView;

})(CellEditorView);

export var SelectEditor = (function(superClass) {
  extend1(SelectEditor, superClass);

  function SelectEditor() {
    return SelectEditor.__super__.constructor.apply(this, arguments);
  }

  SelectEditor.prototype.type = 'SelectEditor';

  SelectEditor.prototype.default_view = SelectEditorView;

  SelectEditor.define({
    options: [p.Array, []]
  });

  return SelectEditor;

})(CellEditor);

export var PercentEditorView = (function(superClass) {
  extend1(PercentEditorView, superClass);

  function PercentEditorView() {
    return PercentEditorView.__super__.constructor.apply(this, arguments);
  }

  return PercentEditorView;

})(CellEditorView);

export var PercentEditor = (function(superClass) {
  extend1(PercentEditor, superClass);

  function PercentEditor() {
    return PercentEditor.__super__.constructor.apply(this, arguments);
  }

  PercentEditor.prototype.type = 'PercentEditor';

  PercentEditor.prototype.default_view = PercentEditorView;

  return PercentEditor;

})(CellEditor);

export var CheckboxEditorView = (function(superClass) {
  extend1(CheckboxEditorView, superClass);

  function CheckboxEditorView() {
    return CheckboxEditorView.__super__.constructor.apply(this, arguments);
  }

  CheckboxEditorView.prototype.input = '<input type="checkbox" value="true" />';

  CheckboxEditorView.prototype.renderEditor = function() {
    return this.focus();
  };

  CheckboxEditorView.prototype.loadValue = function(item) {
    this.defaultValue = !!item[this.args.column.field];
    return this.$input.prop('checked', this.defaultValue);
  };

  CheckboxEditorView.prototype.serializeValue = function() {
    return this.$input.prop('checked');
  };

  return CheckboxEditorView;

})(CellEditorView);

export var CheckboxEditor = (function(superClass) {
  extend1(CheckboxEditor, superClass);

  function CheckboxEditor() {
    return CheckboxEditor.__super__.constructor.apply(this, arguments);
  }

  CheckboxEditor.prototype.type = 'CheckboxEditor';

  CheckboxEditor.prototype.default_view = CheckboxEditorView;

  return CheckboxEditor;

})(CellEditor);

export var IntEditorView = (function(superClass) {
  extend1(IntEditorView, superClass);

  function IntEditorView() {
    return IntEditorView.__super__.constructor.apply(this, arguments);
  }

  IntEditorView.prototype.input = '<input type="text" />';

  IntEditorView.prototype.renderEditor = function() {
    this.$input.spinner({
      step: this.model.step
    });
    return this.$input.focus().select();
  };

  IntEditorView.prototype.remove = function() {
    this.$input.spinner("destroy");
    return IntEditorView.__super__.remove.call(this);
  };

  IntEditorView.prototype.serializeValue = function() {
    return parseInt(this.getValue(), 10) || 0;
  };

  IntEditorView.prototype.loadValue = function(item) {
    IntEditorView.__super__.loadValue.call(this, item);
    this.$input[0].defaultValue = this.defaultValue;
    return this.$input.select();
  };

  IntEditorView.prototype.validateValue = function(value) {
    if (isNaN(value)) {
      return {
        valid: false,
        msg: "Please enter a valid integer"
      };
    } else {
      return IntEditorView.__super__.validateValue.call(this, value);
    }
  };

  return IntEditorView;

})(CellEditorView);

export var IntEditor = (function(superClass) {
  extend1(IntEditor, superClass);

  function IntEditor() {
    return IntEditor.__super__.constructor.apply(this, arguments);
  }

  IntEditor.prototype.type = 'IntEditor';

  IntEditor.prototype.default_view = IntEditorView;

  IntEditor.define({
    step: [p.Number, 1]
  });

  return IntEditor;

})(CellEditor);

export var NumberEditorView = (function(superClass) {
  extend1(NumberEditorView, superClass);

  function NumberEditorView() {
    return NumberEditorView.__super__.constructor.apply(this, arguments);
  }

  NumberEditorView.prototype.input = '<input type="text" />';

  NumberEditorView.prototype.renderEditor = function() {
    this.$input.spinner({
      step: this.model.step
    });
    return this.$input.focus().select();
  };

  NumberEditorView.prototype.remove = function() {
    this.$input.spinner("destroy");
    return NumberEditorView.__super__.remove.call(this);
  };

  NumberEditorView.prototype.serializeValue = function() {
    return parseFloat(this.getValue()) || 0.0;
  };

  NumberEditorView.prototype.loadValue = function(item) {
    NumberEditorView.__super__.loadValue.call(this, item);
    this.$input[0].defaultValue = this.defaultValue;
    return this.$input.select();
  };

  NumberEditorView.prototype.validateValue = function(value) {
    if (isNaN(value)) {
      return {
        valid: false,
        msg: "Please enter a valid number"
      };
    } else {
      return NumberEditorView.__super__.validateValue.call(this, value);
    }
  };

  return NumberEditorView;

})(CellEditorView);

export var NumberEditor = (function(superClass) {
  extend1(NumberEditor, superClass);

  function NumberEditor() {
    return NumberEditor.__super__.constructor.apply(this, arguments);
  }

  NumberEditor.prototype.type = 'NumberEditor';

  NumberEditor.prototype.default_view = NumberEditorView;

  NumberEditor.define({
    step: [p.Number, 0.01]
  });

  return NumberEditor;

})(CellEditor);

export var TimeEditorView = (function(superClass) {
  extend1(TimeEditorView, superClass);

  function TimeEditorView() {
    return TimeEditorView.__super__.constructor.apply(this, arguments);
  }

  return TimeEditorView;

})(CellEditorView);

export var TimeEditor = (function(superClass) {
  extend1(TimeEditor, superClass);

  function TimeEditor() {
    return TimeEditor.__super__.constructor.apply(this, arguments);
  }

  TimeEditor.prototype.type = 'TimeEditor';

  TimeEditor.prototype.default_view = TimeEditorView;

  return TimeEditor;

})(CellEditor);

export var DateEditorView = (function(superClass) {
  extend1(DateEditorView, superClass);

  function DateEditorView() {
    return DateEditorView.__super__.constructor.apply(this, arguments);
  }

  DateEditorView.prototype.emptyValue = new Date();

  DateEditorView.prototype.input = '<input type="text" />';

  DateEditorView.prototype.renderEditor = function() {
    this.calendarOpen = false;
    this.$input.datepicker({
      showOn: "button",
      buttonImageOnly: true,
      beforeShow: (function(_this) {
        return function() {
          return _this.calendarOpen = true;
        };
      })(this),
      onClose: (function(_this) {
        return function() {
          return _this.calendarOpen = false;
        };
      })(this)
    });
    this.$input.siblings(".bk-ui-datepicker-trigger").css({
      "vertical-align": "middle"
    });
    this.$input.width(this.$input.width() - (14 + 2 * 4 + 4));
    return this.$input.focus().select();
  };

  DateEditorView.prototype.destroy = function() {
    $.datepicker.dpDiv.stop(true, true);
    this.$input.datepicker("hide");
    this.$input.datepicker("destroy");
    return DateEditorView.__super__.destroy.call(this);
  };

  DateEditorView.prototype.show = function() {
    if (this.calendarOpen) {
      $.datepicker.dpDiv.stop(true, true).show();
    }
    return DateEditorView.__super__.show.call(this);
  };

  DateEditorView.prototype.hide = function() {
    if (this.calendarOpen) {
      $.datepicker.dpDiv.stop(true, true).hide();
    }
    return DateEditorView.__super__.hide.call(this);
  };

  DateEditorView.prototype.position = function(position) {
    if (this.calendarOpen) {
      $.datepicker.dpDiv.css({
        top: position.top + 30,
        left: position.left
      });
    }
    return DateEditorView.__super__.position.call(this);
  };

  DateEditorView.prototype.getValue = function() {
    return this.$input.datepicker("getDate").getTime();
  };

  DateEditorView.prototype.setValue = function(val) {
    return this.$input.datepicker("setDate", new Date(val));
  };

  return DateEditorView;

})(CellEditorView);

export var DateEditor = (function(superClass) {
  extend1(DateEditor, superClass);

  function DateEditor() {
    return DateEditor.__super__.constructor.apply(this, arguments);
  }

  DateEditor.prototype.type = 'DateEditor';

  DateEditor.prototype.default_view = DateEditorView;

  return DateEditor;

})(CellEditor);
