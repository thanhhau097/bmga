var extend1 = function(child, parent) { for (var key in parent) { if (hasProp.call(parent, key)) child[key] = parent[key]; } function ctor() { this.constructor = child; } ctor.prototype = parent.prototype; child.prototype = new ctor(); child.__super__ = parent.prototype; return child; },
  hasProp = {}.hasOwnProperty,
  slice = [].slice;

import {
  logger
} from "./logging";

import {
  Signal,
  Signalable
} from "./signaling";

import * as property_mixins from "./property_mixins";

import * as refs from "./util/refs";

import * as p from "./properties";

import {
  uniqueId
} from "./util/string";

import {
  max
} from "./util/array";

import {
  extend,
  values,
  clone,
  isEmpty
} from "./util/object";

import {
  isString,
  isObject,
  isArray
} from "./util/types";

import {
  isEqual
} from './util/eq';

export var HasProps = (function() {
  extend1(HasProps.prototype, Signalable);

  HasProps.getters = function(specs) {
    var fn, name, results;
    results = [];
    for (name in specs) {
      fn = specs[name];
      results.push(Object.defineProperty(this.prototype, name, {
        get: fn
      }));
    }
    return results;
  };

  HasProps.prototype.props = {};

  HasProps.prototype.mixins = [];

  HasProps.define = function(object) {
    var name, prop, results;
    results = [];
    for (name in object) {
      prop = object[name];
      results.push((function(_this) {
        return function(name, prop) {
          var default_value, internal, props, refined_prop, type;
          if (_this.prototype.props[name] != null) {
            throw new Error("attempted to redefine property '" + _this.name + "." + name + "'");
          }
          if (_this.prototype[name] != null) {
            throw new Error("attempted to redefine attribute '" + _this.name + "." + name + "'");
          }
          Object.defineProperty(_this.prototype, name, {
            get: function() {
              var value;
              value = this.getv(name);
              return value;
            },
            set: function(value) {
              this.setv(name, value);
              return this;
            }
          }, {
            configurable: false,
            enumerable: true
          });
          type = prop[0], default_value = prop[1], internal = prop[2];
          refined_prop = {
            type: type,
            default_value: default_value,
            internal: internal != null ? internal : false
          };
          props = clone(_this.prototype.props);
          props[name] = refined_prop;
          return _this.prototype.props = props;
        };
      })(this)(name, prop));
    }
    return results;
  };

  HasProps.internal = function(object) {
    var _object, fn1, name, prop;
    _object = {};
    fn1 = (function(_this) {
      return function(name, prop) {
        var default_value, type;
        type = prop[0], default_value = prop[1];
        return _object[name] = [type, default_value, true];
      };
    })(this);
    for (name in object) {
      prop = object[name];
      fn1(name, prop);
    }
    return this.define(_object);
  };

  HasProps.mixin = function() {
    var mixins, names;
    names = 1 <= arguments.length ? slice.call(arguments, 0) : [];
    this.define(property_mixins.create(names));
    mixins = this.prototype.mixins.concat(names);
    return this.prototype.mixins = mixins;
  };

  HasProps.mixins = function(names) {
    return this.mixin.apply(this, names);
  };

  HasProps.override = function(name_or_object, default_value) {
    var name, object, results;
    if (isString(name_or_object)) {
      object = {};
      object[name] = default_value;
    } else {
      object = name_or_object;
    }
    results = [];
    for (name in object) {
      default_value = object[name];
      results.push((function(_this) {
        return function(name, default_value) {
          var props, value;
          value = _this.prototype.props[name];
          if (value == null) {
            throw new Error("attempted to override nonexistent '" + _this.name + "." + name + "'");
          }
          props = clone(_this.prototype.props);
          props[name] = extend({}, value, {
            default_value: default_value
          });
          return _this.prototype.props = props;
        };
      })(this)(name, default_value));
    }
    return results;
  };

  HasProps.define({
    id: [p.Any]
  });

  HasProps.prototype.toString = function() {
    return this.type + "(" + this.id + ")";
  };

  function HasProps(attributes, options) {
    var default_value, name, ref, ref1, type;
    if (attributes == null) {
      attributes = {};
    }
    if (options == null) {
      options = {};
    }
    this.document = null;
    this.destroyed = new Signal(this, "destroyed");
    this.change = new Signal(this, "change");
    this.propchange = new Signal(this, "propchange");
    this.transformchange = new Signal(this, "transformchange");
    this.attributes = {};
    this.properties = {};
    ref = this.props;
    for (name in ref) {
      ref1 = ref[name], type = ref1.type, default_value = ref1.default_value;
      if (type == null) {
        throw new Error("undefined property type for " + this.type + "." + name);
      }
      this.properties[name] = new type({
        obj: this,
        attr: name,
        default_value: default_value
      });
    }
    this._set_after_defaults = {};
    if (attributes.id == null) {
      this.setv("id", uniqueId(), {
        silent: true
      });
    }
    this.setv(attributes, extend({
      silent: true
    }, options));
    if (!options.defer_initialization) {
      this.finalize(attributes, options);
    }
  }

  HasProps.prototype.finalize = function(attributes, options) {
    var name, prop, ref;
    ref = this.properties;
    for (name in ref) {
      prop = ref[name];
      prop.update();
      if (prop.spec.transform) {
        this.connect(prop.spec.transform.change, function() {
          return this.transformchange.emit();
        });
      }
    }
    this.initialize(attributes, options);
    return this.connect_signals();
  };

  HasProps.prototype.initialize = function(attributes, options) {};

  HasProps.prototype.connect_signals = function() {};

  HasProps.prototype.disconnect_signals = function() {
    return Signal.disconnectReceiver(this);
  };

  HasProps.prototype.destroy = function() {
    this.disconnect_signals();
    return this.destroyed.emit();
  };

  HasProps.prototype.clone = function() {
    return new this.constructor(this.attributes);
  };

  HasProps.prototype._setv = function(attrs, options) {
    var attr, changes, changing, current, i, j, ref, silent, val;
    silent = options.silent;
    changes = [];
    changing = this._changing;
    this._changing = true;
    current = this.attributes;
    for (attr in attrs) {
      val = attrs[attr];
      val = attrs[attr];
      if (!isEqual(current[attr], val)) {
        changes.push(attr);
      }
      current[attr] = val;
    }
    if (!silent) {
      if (changes.length) {
        this._pending = true;
      }
      for (i = j = 0, ref = changes.length; 0 <= ref ? j < ref : j > ref; i = 0 <= ref ? ++j : --j) {
        this.properties[changes[i]].change.emit(current[changes[i]]);
      }
    }
    if (changing) {
      return this;
    }
    if (!silent && !options.no_change) {
      while (this._pending) {
        this._pending = false;
        this.change.emit();
      }
    }
    this._pending = false;
    this._changing = false;
    return this;
  };

  HasProps.prototype.setv = function(key, value, options) {
    var attrs, old, prop_name, results, val;
    if (isObject(key) || key === null) {
      attrs = key;
      options = value;
    } else {
      attrs = {};
      attrs[key] = value;
    }
    if (options == null) {
      options = {};
    }
    for (key in attrs) {
      if (!hasProp.call(attrs, key)) continue;
      val = attrs[key];
      prop_name = key;
      if (this.props[prop_name] == null) {
        throw new Error("property " + this.type + "." + prop_name + " wasn't declared");
      }
      if (!((options != null) && options.defaults)) {
        this._set_after_defaults[key] = true;
      }
    }
    if (!isEmpty(attrs)) {
      old = {};
      for (key in attrs) {
        value = attrs[key];
        old[key] = this.getv(key);
      }
      this._setv(attrs, options);
      if ((options != null ? options.silent : void 0) == null) {
        results = [];
        for (key in attrs) {
          value = attrs[key];
          results.push(this._tell_document_about_change(key, old[key], this.getv(key), options));
        }
        return results;
      }
    }
  };

  HasProps.prototype.set = function(key, value, options) {
    logger.warn("HasProps.set('prop_name', value) is deprecated, use HasProps.prop_name = value instead");
    return this.setv(key, value, options);
  };

  HasProps.prototype.get = function(prop_name) {
    logger.warn("HasProps.get('prop_name') is deprecated, use HasProps.prop_name instead");
    return this.getv(prop_name);
  };

  HasProps.prototype.getv = function(prop_name) {
    if (this.props[prop_name] == null) {
      throw new Error("property " + this.type + "." + prop_name + " wasn't declared");
    } else {
      return this.attributes[prop_name];
    }
  };

  HasProps.prototype.ref = function() {
    return refs.create_ref(this);
  };

  HasProps.prototype.set_subtype = function(subtype) {
    return this._subtype = subtype;
  };

  HasProps.prototype.attribute_is_serializable = function(attr) {
    var prop;
    prop = this.props[attr];
    if (prop == null) {
      throw new Error(this.type + ".attribute_is_serializable('" + attr + "'): " + attr + " wasn't declared");
    } else {
      return !prop.internal;
    }
  };

  HasProps.prototype.serializable_attributes = function() {
    var attrs, name, ref, value;
    attrs = {};
    ref = this.attributes;
    for (name in ref) {
      value = ref[name];
      if (this.attribute_is_serializable(name)) {
        attrs[name] = value;
      }
    }
    return attrs;
  };

  HasProps._value_to_json = function(key, value, optional_parent_object) {
    var i, j, len, ref_array, ref_obj, subkey, v;
    if (value instanceof HasProps) {
      return value.ref();
    } else if (isArray(value)) {
      ref_array = [];
      for (i = j = 0, len = value.length; j < len; i = ++j) {
        v = value[i];
        ref_array.push(HasProps._value_to_json(i, v, value));
      }
      return ref_array;
    } else if (isObject(value)) {
      ref_obj = {};
      for (subkey in value) {
        if (!hasProp.call(value, subkey)) continue;
        ref_obj[subkey] = HasProps._value_to_json(subkey, value[subkey], value);
      }
      return ref_obj;
    } else {
      return value;
    }
  };

  HasProps.prototype.attributes_as_json = function(include_defaults, value_to_json) {
    var attrs, key, ref, value;
    if (include_defaults == null) {
      include_defaults = true;
    }
    if (value_to_json == null) {
      value_to_json = HasProps._value_to_json;
    }
    attrs = {};
    ref = this.serializable_attributes();
    for (key in ref) {
      if (!hasProp.call(ref, key)) continue;
      value = ref[key];
      if (include_defaults) {
        attrs[key] = value;
      } else if (key in this._set_after_defaults) {
        attrs[key] = value;
      }
    }
    return value_to_json("attributes", attrs, this);
  };

  HasProps._json_record_references = function(doc, v, result, recurse) {
    var elem, j, k, len, model, results, results1;
    if (v == null) {

    } else if (refs.is_ref(v)) {
      if (!(v.id in result)) {
        model = doc.get_model_by_id(v.id);
        return HasProps._value_record_references(model, result, recurse);
      }
    } else if (isArray(v)) {
      results = [];
      for (j = 0, len = v.length; j < len; j++) {
        elem = v[j];
        results.push(HasProps._json_record_references(doc, elem, result, recurse));
      }
      return results;
    } else if (isObject(v)) {
      results1 = [];
      for (k in v) {
        if (!hasProp.call(v, k)) continue;
        elem = v[k];
        results1.push(HasProps._json_record_references(doc, elem, result, recurse));
      }
      return results1;
    }
  };

  HasProps._value_record_references = function(v, result, recurse) {
    var elem, immediate, j, k, l, len, len1, obj, results, results1, results2;
    if (v == null) {

    } else if (v instanceof HasProps) {
      if (!(v.id in result)) {
        result[v.id] = v;
        if (recurse) {
          immediate = v._immediate_references();
          results = [];
          for (j = 0, len = immediate.length; j < len; j++) {
            obj = immediate[j];
            results.push(HasProps._value_record_references(obj, result, true));
          }
          return results;
        }
      }
    } else if (v.buffer instanceof ArrayBuffer) {

    } else if (isArray(v)) {
      results1 = [];
      for (l = 0, len1 = v.length; l < len1; l++) {
        elem = v[l];
        results1.push(HasProps._value_record_references(elem, result, recurse));
      }
      return results1;
    } else if (isObject(v)) {
      results2 = [];
      for (k in v) {
        if (!hasProp.call(v, k)) continue;
        elem = v[k];
        results2.push(HasProps._value_record_references(elem, result, recurse));
      }
      return results2;
    }
  };

  HasProps.prototype._immediate_references = function() {
    var attrs, key, result, value;
    result = {};
    attrs = this.serializable_attributes();
    for (key in attrs) {
      value = attrs[key];
      HasProps._value_record_references(value, result, false);
    }
    return values(result);
  };

  HasProps.prototype.references = function() {
    var references;
    references = {};
    HasProps._value_record_references(this, references, true);
    return values(references);
  };

  HasProps.prototype.attach_document = function(doc) {
    if (this.document !== null && this.document !== doc) {
      throw new Error("models must be owned by only a single document");
    }
    this.document = doc;
    if (this._doc_attached != null) {
      return this._doc_attached();
    }
  };

  HasProps.prototype.detach_document = function() {
    return this.document = null;
  };

  HasProps.prototype._tell_document_about_change = function(attr, old, new_, options) {
    var need_invalidate, new_id, new_ref, new_refs, old_id, old_ref, old_refs;
    if (!this.attribute_is_serializable(attr)) {
      return;
    }
    if (this.document !== null) {
      new_refs = {};
      HasProps._value_record_references(new_, new_refs, false);
      old_refs = {};
      HasProps._value_record_references(old, old_refs, false);
      need_invalidate = false;
      for (new_id in new_refs) {
        new_ref = new_refs[new_id];
        if (!(new_id in old_refs)) {
          need_invalidate = true;
          break;
        }
      }
      if (!need_invalidate) {
        for (old_id in old_refs) {
          old_ref = old_refs[old_id];
          if (!(old_id in new_refs)) {
            need_invalidate = true;
            break;
          }
        }
      }
      if (need_invalidate) {
        this.document._invalidate_all_models();
      }
      return this.document._notify_change(this, attr, old, new_, options);
    }
  };

  HasProps.prototype.materialize_dataspecs = function(source) {
    var data, name, prop, ref;
    data = {};
    ref = this.properties;
    for (name in ref) {
      prop = ref[name];
      if (!prop.dataspec) {
        continue;
      }
      if ((prop.optional || false) && prop.spec.value === null && (!(name in this._set_after_defaults))) {
        continue;
      }
      data["_" + name] = prop.array(source);
      if ((prop.spec.field != null) && prop.spec.field in source._shapes) {
        data["_" + name + "_shape"] = source._shapes[prop.spec.field];
      }
      if (prop instanceof p.Distance) {
        data["max_" + name] = max(data["_" + name]);
      }
    }
    return data;
  };

  return HasProps;

})();
