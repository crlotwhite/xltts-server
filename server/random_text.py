def get_random_key():
    s = ''
    gen = _random_char_generator()
    for _ in range(16):
        s += next(gen)

    return s


def _random_char_generator():
    from random import choice
    _char = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    while True:
        yield choice(_char)
