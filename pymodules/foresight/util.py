# util.py


def fullname(o):
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__
    return module + '.' + o.__class__.__name__


def ValidateType(obj,
                 reqd_type=None,
                 val_func=None,
                 err_msg='',
                 arg_name='',
                 allow_none=False):
    if not allow_none:
        if (obj is None): raise TypeError(arg_name + ' cannot be None')

    # User cannot supply both a type name and a validation function
    # assert: only 1 can have the value
    assert (((reqd_type is None) and (val_func is not None))
            or ((reqd_type is not None) and (val_func is None)))
    assert (not ((reqd_type is None) and (val_func is None)))

    if obj is not None:
        if reqd_type is not None:
            if (not isinstance(obj, reqd_type)):
                raise TypeError(arg_name + ' ' + err_msg)

        else:
            if (not val_func(obj)):
                raise TypeError(arg_name + ' ' + err_msg)

def round_down(x, base=5):
    return base * (round(x/base) - 1)

def VarExists(x):
    try: x
    except NameError:
        return False
    return True