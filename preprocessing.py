import utilities

def get_funcs():
    return IMG_FUNC

# processa nota de 2 reais 
def process_2front(img):
    return 

def process_2back(img):
    
    img_result = utilities.adaptive_thresholding(img)
    
    return img_result

# **********************************

# processa nota de 5 reais
def process_5front(img):
    return

def process_5back(img):
    return 

# **********************************

# processa nota de 10 reais
def process_10front(img):
    return
 
def process_10back(img):
    return 

# **********************************

# processa nota de 20 reais
def process_20front(img):
    return 

def process_20back(img):
    return
# **********************************

# processa nota de 50 reais
def process_50front(img):
    return 

def process_50back(img):
    return
# **********************************

# processa nota de 20 reais
def process_100front(img):
    return 

def process_100back(img):
    return
# **********************************

IMG_FUNC = {
    "bill_scans/2_back.jpg": process_2back,
    "bill_scans/2_front.jpg": process_2front,
    "bill_scans/5_back.jpg": process_5back,
    "bill_scans/5_front.jpg": process_5front,
    "bill_scans/10_back.jpg": process_10back,
    "bill_scans/10_front.jpg": process_10front,
    "bill_scans/20_back.jpg": process_20back,
    "bill_scans/20_front.jpg": process_20front,
    "bill_scans/50_back.jpg": process_50back,
    "bill_scans/50_front.jpg": process_50front,
    "bill_scans/100_back.jpg": process_100back,
    "bill_scans/100_front.jpg": process_100front
}
