from __future__ import annotations

from functools import partial

from helpers.attribute_handler_functions import snake, parse_parameter_value, parse_temperature_value, parse_write_cycle_time


PARAMETER_HANDLERS = {
    # Basic snake handlers
    "basic": {
        110: (snake, None),
        248: (snake, None),
        74: (snake, None),
        1977: (snake, None),
        2700: (snake, None),
        69: (snake, None),
        405: (snake, None),
        2697: (snake, None),
        149: (snake, None),
        961: (snake, None),
        570: (snake, None),
        2043: (snake, None),
        183: (snake, None),
        578: (snake, None),
        1684: (snake, None),
    },

    # Parameter value handlers
    "param_value": {
        892: ("transition_frequency", parse_parameter_value),
        1176: ("noise_figure", parse_parameter_value),
        445: (snake, parse_parameter_value),
        2095: (snake, parse_parameter_value),
        800: (snake, parse_parameter_value),
        2192: (snake, parse_parameter_value),
        1683: (snake, parse_parameter_value),
    },

    # Comma-blocked parameter handlers
    "comma_blocked": {
        814: ("collector_emitter_breakdown_max", partial(parse_parameter_value, block=',')),
        588: (snake, partial(parse_parameter_value, block=',')),
        812: (snake, partial(parse_parameter_value, block=',')),
        127: (snake, partial(parse_parameter_value, block=',')),
        276: (snake, partial(parse_parameter_value, block=',')),
        142: (snake, partial(parse_parameter_value, block=',')),
        2689: (snake, partial(parse_parameter_value, block=',')),
        404: (snake, partial(parse_parameter_value, block=',')),
        448: (snake, partial(parse_parameter_value, block=',')),
    },

    # Special cases
    "special": {
        891: ('dc_current_gain_min', partial(parse_parameter_value, block='/', defined='hFE')),
        252: ('operating_temperature', parse_temperature_value),
        2042: ("write_cycle_time", parse_write_cycle_time)
    }
}


# Combine all handlers into a single dictionary
def get_handler_dict():
    return {
        id_: handler
        for group in PARAMETER_HANDLERS.values()
        for id_, handler in group.items()
    }



fndict = get_handler_dict()
