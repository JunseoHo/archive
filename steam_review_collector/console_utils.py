"""
    ANSI Code
"""
import sys

ANSI_CODES = {
    'white': '\033[37m',
    'blue': '\033[34m',
    'green': '\033[32m',
    'no_color': '\033[0m'
}


def print_console(text, color="white"):
    print(ANSI_CODES[color], end='')
    print(text, end='')
    print(ANSI_CODES['no_color'])

