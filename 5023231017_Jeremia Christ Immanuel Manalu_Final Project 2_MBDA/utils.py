# utils.py
def fix_orientation(img):
    return img.permute(0, 2, 1)

def adjust_label(y):
    return y - 1