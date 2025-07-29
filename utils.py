def stars(text):
    if len(text) == 0:
        return '*' * 80
    t = (60 - len(text)) // 2
    return f"{'*' * t}{' ' * 10}{text}{' ' * 10}{'*' * t}"

def hashtags(text):
    t = (60 - len(text)) // 2
    return f"{'#' * t}{' ' * 10}{text}{' ' * 10}{'#' * t}"