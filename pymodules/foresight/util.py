# util.py
# util.py

def fullname(o):
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__
    return module + '.' + o.__class__.__name__

def ValidateType(obj, reqd_type, err_msg = '', arg_name = '', allow_none = False):
    if not allow_none:
        if (obj is None): raise TypeError(arg_name + ' cannot be None')
    if (not isinstance(obj, reqd_type)):
            raise TypeError(arg_name + ' ' + err_msg)
