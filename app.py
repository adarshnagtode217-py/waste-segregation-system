# import streamlit as st
# import numpy as np
# from PIL import Image
# from tensorflow.keras.models import load_model

# # Load Model
# model = load_model("waste_classifier_mobilenet.h5")

# # Classes
# classes = [
#     'cardboard',
#     'glass',
#     'metal',
#     'paper',
#     'plastic',
#     'trash'
# ]

# # Bin Suggestions
# bins = {
#     'cardboard': '🟢 Green Bin',
#     'glass': '⚪ White Bin',
#     'metal': '🟡 Yellow Bin',
#     'paper': '🟢 Green Bin',
#     'plastic': '🔵 Blue Bin',
#     'trash': '⚫ Black Bin'
# }

# # Page Config
# st.set_page_config(
#     page_title="Waste Segregation System",
#     page_icon="♻️",
#     layout="centered"
# )

# st.title("♻️ AI Waste Segregation System")

# st.write("Upload an image and the AI model will identify the waste category.")

# uploaded_file = st.file_uploader(
#     "Choose an Image",
#     type=["jpg", "jpeg", "png"]
# )

# if uploaded_file is not None:

#     image = Image.open(uploaded_file)

#     st.image(image, caption="Uploaded Image", use_container_width=True)

#     img = image.resize((224,224))

#     img = np.array(img)

#     # Handle grayscale images
#     if len(img.shape) == 2:
#         img = np.stack((img,)*3, axis=-1)

#     # Remove alpha channel if present
#     if img.shape[-1] == 4:
#         img = img[:, :, :3]

#     img = img / 255.0

#     img = np.expand_dims(img, axis=0)

#     prediction = model.predict(img)

#     index = np.argmax(prediction)

#     waste = classes[index]

#     confidence = np.max(prediction) * 100

#     st.success(f"Predicted Waste: {waste.upper()}")

#     st.info(f"Confidence: {confidence:.2f}%")

#     st.warning(f"Suggested Bin: {bins[waste]}")

#     st.subheader("Prediction Probabilities")

#     for i in range(len(classes)):
#         st.write(
#             f"{classes[i]} : {prediction[0][i]*100:.2f}%"
#         )
#         st.progress(float(prediction[0][i]))

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


model = load_model("waste_classifier_mobilenet.h5")


classes=[
'cardboard',
'glass',
'metal',
'paper',
'plastic',
'trash'
]


bins={

'cardboard':'🟢 Green Bin',

'glass':'⚪ White Bin',

'metal':'🟡 Yellow Bin',

'paper':'🟢 Green Bin',

'plastic':'🔵 Blue Bin',

'trash':'⚫ Black Bin'

}


st.set_page_config(

page_title="AI Waste Segregation",

page_icon="♻️",

layout="wide"

)



st.markdown("""

<style>


.stApp{

background:linear-gradient(135deg,#00140d,#00291d,#00331f);

color:white;

}



header{

visibility:hidden;

}


footer{

visibility:hidden;

}





.title{

font-size:65px;

font-weight:700;

text-align:center;

margin-top:30px;

color:white;

}



.green{

color:#8CFF00;

}



.subtitle{

text-align:center;

font-size:24px;

color:#d0d0d0;

margin-bottom:50px;

}




[data-testid="stFileUploader"]{


border:2px solid #75ff00;


border-radius:20px;


padding:20px;


background:rgba(255,255,255,0.04);


backdrop-filter:blur(8px);


box-shadow:0px 0px 25px rgba(117,255,0,0.3);

}





.predcard{


padding:30px;


border-radius:20px;


background:rgba(255,255,255,0.05);


box-shadow:0px 0px 30px rgba(0,255,100,0.4);


text-align:center;


margin-top:30px;

}




.feature{


padding:20px;


border-radius:15px;


background:rgba(255,255,255,0.04);


text-align:center;


box-shadow:0px 0px 20px rgba(0,255,100,0.2);

}



</style>


""",unsafe_allow_html=True)






st.markdown("""

<h1 class='title'>

♻️ AI Waste Segregation

<span class='green'>

System

</span>

</h1>


""",unsafe_allow_html=True)






st.markdown("""

<p class='subtitle'>

Upload an image and the AI model will identify the waste category


</p>


""",unsafe_allow_html=True)





uploaded_file=st.file_uploader(

"Choose an Image",

type=['jpg','jpeg','png']

)





if uploaded_file:



    image=Image.open(uploaded_file)



    st.image(

    image,

    width=300

    )



    img=image.resize(

    (224,224)

    )



    img=np.array(img)



    if len(img.shape)==2:


        img=np.stack(

        (img,)*3,

        axis=-1

        )



    if img.shape[-1]==4:


        img=img[:,:,:3]




    img=img/255.0




    img=np.expand_dims(

    img,

    axis=0

    )




    prediction=model.predict(img)



    index=np.argmax(prediction)



    waste=classes[index]



    confidence=np.max(prediction)*100





    st.markdown(f"""


<div class='predcard'>


<h1>

{waste.upper()}

</h1>



<h3>

Confidence :

{confidence:.2f}%


</h3>



<h2>

{bins[waste]}

</h2>



</div>


""",unsafe_allow_html=True)






    st.subheader(

    "📊 Prediction Probabilities"

    )




    for i in range(len(classes)):



        st.write(


        f"{classes[i]} : {prediction[0][i]*100:.2f}%"

        )



        st.progress(

        float(prediction[0][i])

        )






st.markdown("<br>",unsafe_allow_html=True)



c1,c2,c3=st.columns(3)



with c1:


    st.markdown("""


<div class='feature'>


<h3>

🛡 AI Powered


</h3>


MobileNetV2


</div>



""",unsafe_allow_html=True)






with c2:



    st.markdown("""


<div class='feature'>


<h3>

🎯 High Accuracy


</h3>


Smart Classification


</div>



""",unsafe_allow_html=True)







with c3:


    st.markdown("""


<div class='feature'>


<h3>

🌿 Eco Friendly


</h3>


Better Environment


</div>



""",unsafe_allow_html=True)