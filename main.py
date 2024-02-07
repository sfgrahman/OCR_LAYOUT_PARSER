import streamlit as st
import layoutparser as lp
from PIL import Image
import numpy as np
import cv2
import keras_ocr
import math
import re
try:
    from detectron2.config import get_cfg  
except ModuleNotFoundError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'git+https://github.com/facebookresearch/detectron2.git'])


model = lp.Detectron2LayoutModel('lp://TableBank/faster_rcnn_R_101_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.3],
                                 label_map={0:"Table"})


def get_distance(predictions):
    """
    Function returns dictionary with (key,value):
        * text : detected text in image
        * center_x : center of bounding box (x)
        * center_y : center of bounding box (y)
        * distance_from_origin : hypotenuse
        * distance_y : distance between y and origin (0,0)
    """
    
    # Point of origin
    x0, y0 = 0, 0
    # Generate dictionary
    detections = []
    for group in predictions:
        # Get center point of bounding box
        top_left_x, top_left_y = group[1][0]
        bottom_right_x, bottom_right_y = group[1][1]
        center_x = (top_left_x + bottom_right_x) / 2
        center_y = (top_left_y + bottom_right_y) / 2
    # Use the Pythagorean Theorem to solve for distance from origin
    distance_from_origin = math.dist([x0,y0], [center_x, center_y])
    # Calculate difference between y and origin to get unique rows
    distance_y = center_y - y0
    # Append all results
    detections.append({
                        'text':group[0],
                        'center_x':center_x,
                        'center_y':center_y,
                        'distance_from_origin':distance_from_origin,
                        'distance_y':distance_y
                    })
    return detections


def distinguish_rows(lst, thresh=15):
    """Function to help distinguish unique rows"""
    
    sublists = [] 
    for i in range(0, len(lst)-1):
        if lst[i+1]['distance_y'] - lst[i]['distance_y'] <= thresh:
            if lst[i] not in sublists:
                sublists.append(lst[i])
            sublists.append(lst[i+1])
        else:
            yield sublists
            sublists = [lst[i+1]]
    yield sublists

def create_opencv_image_from_stringio(img_stream, cv2_img_flag=1):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)


def imageOrder(image, text_blocks):
    h, w = image.shape[:2]

    left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image)

    left_blocks = text_blocks.filter_by(left_interval, center=True)
    left_blocks.sort(key = lambda b:b.coordinates[1])

    right_blocks = [b for b in text_blocks if b not in left_blocks]
    right_blocks.sort(key = lambda b:b.coordinates[1])

    # And finally combine the two list and add the index
    # according to the order
    text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])
    return text_blocks

def ocrAgent(text_output):
    ocr_agent = lp.TesseractAgent(languages='eng')
    # Initialize the tesseract ocr engine. You might need
    # to install the OCR components in layoutparser:
    # pip install layoutparser[ocr]
    for block in text_output:
        segment_image = (block
                        .pad(left=5, right=5, top=5, bottom=5)
                        .crop_image(image))
            # add padding in each image segment can help
            # improve robustness

        text = ocr_agent.detect(segment_image)
        block.set(text=text, inplace=True)
    final_value=[]
    for txt in text_output.get_texts():
        # final_value.append(txt)
        text = re.sub('[\n]+', '\n', txt)
        for sts in text.strip().split('\n'):
            st.write(sts)
          

def kerasOcr(image_path):
    pipeline = keras_ocr.pipeline.Pipeline()
    read_image = keras_ocr.tools.read(image_path)
    prediction_groups = pipeline.recognize([read_image])
    return prediction_groups[0]
    

st.title("Table Detection using Layout Parser")

st.markdown('''
            <p style="text-align:center; color:green; font-size:22px;"><i>Accurate Table Layout Detection with the full power of Deep Learning.</i></p>''', unsafe_allow_html=True)

user_input = st.file_uploader("**Upload your image with table**", type=["png", "jpg", "jpeg"])
#st.write(user_input)
submit = st.button("Submit")
if submit:
    with st.spinner("wait for result .."):
        #image = Image.open(user_input)
        #image_arr = np.array(image)
        #image = image[..., ::-1]
        open_cv_image = create_opencv_image_from_stringio(user_input)
        image = open_cv_image[..., ::-1]
        layout = model.detect(image)
        #st.write(layout)
        text_blocks = lp.Layout([b for b in layout if b.type=='Table'])
        #x1=int(text_blocks[0].block.x_1)
        #y1=int(text_blocks[0].block.y_1)
        #x2=int(text_blocks[0].block.x_2)
        #y2=int(text_blocks[0].block.y_2)
        #img = np.array(Image.open(image))
        #cropping
        #cropped_image = image[y1:y2, x1:x2]
        #cropped_image = Image.fromarray(cropped_image)
        text_output = imageOrder(image, text_blocks)
        ocrAgent(text_output)
        #predictions = kerasOcr(cropped_image)
        #predictions = get_distance(predictions)
        #predictions = list(distinguish_rows(predictions, thresh=15))
        #st.write(predictions)
        
        