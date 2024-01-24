def get_color(rgb: tuple) -> tuple:
    '''Return a color that opencv understand. RGB => BGR'''
    r, g, b = rgb
    return (b, g, r)

def render_text(cv: object, img: object, string: str, position: tuple, font: object, fontSize: int, color: tuple, thickness: int):
    '''Render text with cv2'''
    cv.putText(img, string, position, font, fontSize, color, thickness)