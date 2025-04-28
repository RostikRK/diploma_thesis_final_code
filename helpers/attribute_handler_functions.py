import re


def snake(s):
    """
    Convert a string to the snake case.
    """
    return '_'.join(
        re.sub(r'([A-Z][a-z]+)', r' \1',
               re.sub(r'([A-Z]+)', r' \1',
                      re.sub(r'[-?,\\/()]+', ' ', s)
                      )).split()).lower()


def parse_parameter_value(param_name, value_string, block='', defined=''):
    """
    Parse a parameter value string with various formats.
    """
    # Check for invalid characters or empty value
    if any(char in block for char in value_string):
        return False

    if value_string == '-':
        return None

    result = []

    # Split by @ to separate values from test conditions
    parts = value_string.split('@')

    # Extract values part
    if len(parts) == 1:
        values_part = parts[0].strip()
        test_conditions = [""]
    else:
        values_part = parts[0].strip()
        test_conditions_part = '@'.join(parts[1:]).strip()

        # Parse test conditions
        if '~' in test_conditions_part:
            test_conditions = [tc.strip() for tc in test_conditions_part.split('~')]
        else:
            test_conditions = [test_conditions_part.strip()]

    # Parse values
    if '~' in values_part:
        # Range of values
        value_ranges = [v.strip() for v in values_part.split('~')]

        value_data = []
        for v in value_ranges:
            numeric_part = ''
            unit_part = ''

            for char in v:
                if char.isdigit() or char == '.':
                    numeric_part += char
                else:
                    unit_part += char

            if unit_part.strip() == "":
                unit_part = defined

            value_data.append((numeric_part, unit_part.strip()))

        # Handle min and max values
        min_value, min_unit = value_data[0]
        max_value, max_unit = value_data[-1]

        if len(test_conditions) == 1:
            result.append((f"{param_name}_min", min_value, min_unit, test_conditions[0]))
            result.append((f"{param_name}_max", max_value, max_unit, test_conditions[0]))
        else:
            for i, (value, unit) in enumerate(value_data):
                suffix = "_min" if i == 0 else "_max"
                test_condition = test_conditions[min(i, len(test_conditions) - 1)]
                result.append((f"{param_name}{suffix}", value, unit, test_condition))
    else:
        # Single value
        numeric_part = ''
        unit_part = ''

        for char in values_part:
            if char.isdigit() or char == '.':
                numeric_part += char
            else:
                unit_part += char

        if unit_part.strip() == "":
            unit_part = defined

        result.append((param_name, numeric_part, unit_part.strip(), test_conditions[0]))

    return result


def parse_temperature_value(param_name, value_string, block=',\\', defined='°C'):
    """
    Parse a temperature value string with various formats.
    Handles special notations like TJ and TA.
    """
    # Check for invalid characters or empty value
    if any(char in block for char in value_string):
        return False

    if value_string == '-':
        return None

    result = []

    # Extract temperature type (TJ or TA)
    temp_type = ""
    if "(" in value_string and ")" in value_string:
        start_idx = value_string.find("(")
        end_idx = value_string.find(")")
        temp_type = value_string[start_idx + 1:end_idx].strip()
        value_string = value_string[:start_idx].strip() + value_string[end_idx + 1:].strip()

    # Parse values
    if '~' in value_string:
        # Range of temperatures
        temp_ranges = [v.strip() for v in value_string.split('~')]

        # Extract numeric values and units
        temp_data = []
        for v in temp_ranges:
            numeric_part = ''
            unit_part = ''

            for char in v:
                if char.isdigit() or char == '.' or char == '-':
                    numeric_part += char
                else:
                    unit_part += char

            if unit_part.strip() == "":
                unit_part = defined

            temp_data.append((numeric_part, unit_part.strip()))

        min_value, min_unit = temp_data[0]
        max_value, max_unit = temp_data[-1]

        # Create parameter name with temperature type
        min_param = f"{param_name}_min"
        max_param = f"{param_name}_max"

        if temp_type:
            min_param = f"{min_param}_{temp_type.lower()}"
            max_param = f"{max_param}_{temp_type.lower()}"

        result.append((min_param, min_value, min_unit, temp_type))
        result.append((max_param, max_value, max_unit, temp_type))
    else:
        # Single temperature
        numeric_part = ''
        unit_part = ''

        for char in value_string:
            if char.isdigit() or char == '.' or char == '-':
                numeric_part += char
            else:
                unit_part += char

        if unit_part.strip() == "":
            unit_part = defined

        param = param_name
        if temp_type:
            param = f"{param}_{temp_type.lower()}"

        result.append((param, numeric_part, unit_part.strip(), temp_type))

    return result


def parse_write_cycle_time(param_name, value_string, block='\\', defined='ms'):
    """
    Parse a write cycle time value string with various formats.
    Handles both single values (for word) and comma-separated values (for word and page).
    """
    # Check for invalid characters or empty value
    if any(char in block for char in value_string):
        return False

    if value_string == '-':
        return None

    result = []

    if ',' in value_string:
        # Split for word and page values
        values = [v.strip() for v in value_string.split(',')]

        word_value = values[0]
        numeric_part = ''
        unit_part = ''

        for char in word_value:
            if char.isdigit() or char == '.' or char == '-':
                numeric_part += char
            else:
                unit_part += char

        if unit_part.strip() == "":
            unit_part = defined

        result.append((f"{param_name}_word", numeric_part, unit_part.strip(), ""))

        if len(values) > 1:
            page_value = values[1]
            numeric_part = ''
            unit_part = ''

            for char in page_value:
                if char.isdigit() or char == '.' or char == '-':
                    numeric_part += char
                else:
                    unit_part += char

            if unit_part.strip() == "":
                unit_part = defined

            result.append((f"{param_name}_page", numeric_part, unit_part.strip(), ""))
    else:
        # Process single word values
        numeric_part = ''
        unit_part = ''

        for char in value_string:
            if char.isdigit() or char == '.' or char == '-':
                numeric_part += char
            else:
                unit_part += char

        if unit_part.strip() == "":
            unit_part = defined

        result.append((f"{param_name}_word", numeric_part, unit_part.strip(), ""))

    return result
