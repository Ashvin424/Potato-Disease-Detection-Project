
# import streamlit as st
# import numpy as np
# import tensorflow as tf

# def model_prediction(test_image):
#     model = tf.keras.models.load_model("trained_plant_desiese_model.keras")
#     image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
#     input_arr = tf.keras.preprocessing.image.img_to_array(image)
#     input_arr = np.array([input_arr]) # Convert single image to a batch.
#     predictions = model.predict(input_arr) #runs the model prediction on the image
#     return np.argmax(predictions)

# st.sidebar.title("Plant Disease System for Sustainable Agriculture")
# app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition"])

# # from PIL import Image
# # img = Image.open("images/image.jpg")
# # st.image(img, width=700)

# if(app_mode == "Home"):
#     st.markdown("<h1 style='text-align: center;'>Plant Disease Detection system for Sustainable Agriculture</h1>", unsafe_allow_html=True)

# elif(app_mode == "Disease Recognition"):
#     st.header("Plant Disease Detection System For Sustainable Agriculture")

#     test_image = st.file_uploader("Choose an Image :")
#     if(st.button("Show Image")):
#         st.image(test_image, width = 4, use_column_width=True)

#     if(st.button("Predict")):
#         st.snow()
#         st.write("Our Prediction")
#         result_index = model_prediction(test_image)

#         class_name =['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
#         st.success("Model is predicted its a {}".format(class_name[result_index]))
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

def model_prediction(image_data):
    model = tf.keras.models.load_model("trained_plant_desiese_model.keras")
    image = Image.open(image_data).resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

st.sidebar.title("Plant Disease System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition"])

if app_mode == "Home":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

elif app_mode == "Disease Recognition":
    st.header("Plant Disease Detection System")

    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                result_index = model_prediction(test_image)
                class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
                st.success(f"Model Prediction: {class_name[result_index]}")
                st.balloons()

