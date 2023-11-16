def printt(text, color='w'):
    """
    Custom print function
    :param text: Text to print
    :param color: Color of the text
    :return: None
    """
    # Colors: red, green, blue, yellow
    colors = {
        'r': '\033[31m',
        'g': '\033[32m',
        'b': '\033[34m',
        'y': '\033[33m'
    }
    end_color = '\033[0m'
    if color.lower() in colors:
        print(f"{colors[color.lower()]}{text}{end_color}")
    else:
        print(text)