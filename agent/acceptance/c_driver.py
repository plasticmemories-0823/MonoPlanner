"""One fixed consumer driver, sent into the task backend; never repairs producer output."""

DRIVER_SOURCE = r'''
def build_c_driver(check, raw):
    parameters = ', '.join('double ' + name for name in check['variables']) or 'void'
    if check['kind'] == 'c_expression':
        body = 'return (\n' + raw + '\n);'
    elif check['kind'] == 'c_statement':
        result = check['result_variable']
        # A missing assignment must not accidentally pass a zero-valued test.
        body = 'double ' + result + ' = NAN;\n' + raw + '\nreturn ' + result + ';'
    else:
        raise ValueError('Unsupported raw C consumer')
    calls = '\n'.join('printf("%.17g\\n", agent_value(' + ','.join(
        repr(float(v)) for v in case['values']) + '));' for case in check['cases'])
    return ('#include <math.h>\n#include <stdio.h>\n'
            + 'double agent_value(' + parameters + ') {\n' + body + '\n}\n'
            + 'int main(void) {\n' + calls + '\nreturn 0;\n}\n')
'''


def build_c_driver(check, raw):
    """Use the identical driver locally in tests and in the isolated backend."""
    scope = {}
    exec(DRIVER_SOURCE, scope)
    return scope['build_c_driver'](check, raw)
