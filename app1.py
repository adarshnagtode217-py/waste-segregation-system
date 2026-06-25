import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


st.set_page_config(
    page_title="Smart Agriculture ML System",
    page_icon="🌾",
    layout="wide"
)


st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}

[data-testid="stSidebar"]{
background:#0B1320;
}

[data-testid="stMetric"]{
background:rgba(255,255,255,0.08);
padding:20px;
border-radius:15px;
box-shadow:0px 0px 15px rgba(0,255,0,0.3);
text-align:center;
}

div[data-testid="stPlotlyChart"]{
background:rgba(255,255,255,0.05);
padding:15px;
border-radius:15px;
}

h1,h2,h3{
color:white;
}

p{
color:white;
}

</style>
""",unsafe_allow_html=True)



st.markdown("""
<h1 style='text-align:center;color:#90EE90;'>
🌾 Smart Agriculture ML System
</h1>

<p style='text-align:center;font-size:22px;'>
Crop Recommendation • Fertilizer Recommendation • Yield Prediction
</p>
""",unsafe_allow_html=True)



np.random.seed(42)

n = 500


N = np.random.randint(0,140,n)

P = np.random.randint(5,145,n)

K = np.random.randint(5,205,n)

temperature = np.random.uniform(10,40,n)

humidity = np.random.uniform(20,100,n)

ph = np.random.uniform(4,9,n)

rainfall = np.random.uniform(20,300,n)



crop = np.where(
rainfall>200,
"Rice",

np.where(
ph>7,
"Cotton",

np.where(
temperature>30,
"Maize",
"Wheat"
)

)

)



fertilizer = np.where(

N<80,

"Urea",

np.where(

P<80,

"DAP",

"NPK"

)

)



yield_data=(

N*0.05+

P*0.04+

K*0.03+

rainfall*0.02

)



df=pd.DataFrame({

"N":N,

"P":P,

"K":K,

"Temperature":temperature,

"Humidity":humidity,

"pH":ph,

"Rainfall":rainfall,

"Crop":crop,

"Fertilizer":fertilizer,

"Yield":yield_data

})



crop_map={

v:i

for i,v in enumerate(df["Crop"].unique())

}



fert_map={

v:i

for i,v in enumerate(df["Fertilizer"].unique())

}



df["CropEncoded"]=df["Crop"].map(crop_map)

df["FertilizerEncoded"]=df["Fertilizer"].map(fert_map)



features=[

"N",

"P",

"K",

"Temperature",

"Humidity",

"pH",

"Rainfall"

]


X=df[features]

X_train,X_test,y_train,y_test=train_test_split(
X,
df["CropEncoded"],
test_size=0.2,
random_state=42
)

crop_model=RandomForestClassifier()

crop_model.fit(
X_train,
y_train
)

crop_acc=accuracy_score(
y_test,
crop_model.predict(X_test)
)


fert_model=DecisionTreeClassifier()

fert_model.fit(
X,
df["FertilizerEncoded"]
)



yield_model=RandomForestRegressor()

yield_model.fit(
X,
df["Yield"]
)



st.sidebar.header("🌱Enter Soil Values")



n_val=st.sidebar.slider(
"Nitrogen",
0,
140,
50
)

p_val=st.sidebar.slider(
"Phosphorus",
0,
145,
50
)

k_val=st.sidebar.slider(
"Potassium",
0,
205,
50
)

temp_val=st.sidebar.slider(
"Temperature",
10,
40,
25
)

hum_val=st.sidebar.slider(
"Humidity",
20,
100,
60
)

ph_val=st.sidebar.slider(
"pH",
4.0,
9.0,
6.5
)

rain_val=st.sidebar.slider(
"Rainfall",
20,
300,
150
)



sample=[[
n_val,
p_val,
k_val,
temp_val,
hum_val,
ph_val,
rain_val
]]



crop_pred=crop_model.predict(sample)[0]

fert_pred=fert_model.predict(sample)[0]

yield_pred=yield_model.predict(sample)[0]



crop_name=list(crop_map.keys())[
list(crop_map.values()).index(crop_pred)
]


fert_name=list(fert_map.keys())[
list(fert_map.values()).index(fert_pred)
]

# col1,col2,col3=st.columns(3)

# with col1:
#     st.metric(
#     "Crop Model Accuracy",
#     f"{crop_acc*100:.2f}%"
#     )

# with col2:
#     st.metric(
#     "Recommended Crop",
#     crop_name
#     )

# with col3:
#     st.metric(
#     "Predicted Yield",
#     f"{yield_pred:.2f}"
#     )


# st.markdown(f"""
# <div style="
# background:#1b4332;
# padding:20px;
# border-radius:15px;
# text-align:center;
# margin-top:15px;
# box-shadow:0px 0px 15px rgba(0,255,0,0.4);
# ">

# <h2 style="color:white;">
# 🧪{fert_name}
# </h2>

# <p style="color:#90EE90;font-size:20px;">
# Recommended Fertilizer
# </p>

# </div>
# """,unsafe_allow_html=True)


# st.markdown("---")


# fig1=px.scatter(
# df,
# x="Rainfall",
# y="Yield",
# color="Crop",
# title="Crop Yield Analysis"
# )

# fig1.update_layout(
# template="plotly_dark"
# )

# st.plotly_chart(
# fig1,
# use_container_width=True
# )



# fig2=px.histogram(
# df,
# x="Crop",
# title="Crop Distribution"
# )

# fig2.update_layout(
# template="plotly_dark"
# )

# st.plotly_chart(
# fig2,
# use_container_width=True
# )



# fig3=px.box(
# df,
# x="Crop",
# y="Rainfall",
# color="Crop",
# title="Rainfall Analysis"
# )

# fig3.update_layout(
# template="plotly_dark"
# )

# st.plotly_chart(
# fig3,
# use_container_width=True
# )



# st.subheader("📋Dataset Preview")

# st.dataframe(
# df.head(20),
# use_container_width=True
# )

col1,col2,col3=st.columns(3)

with col1:
    st.metric(
    "Crop Model Accuracy",
    f"{crop_acc*100:.2f}%"
    )

with col2:
    st.metric(
    "Recommended Crop",
    crop_name
    )

with col3:
    st.metric(
    "Predicted Yield",
    f"{yield_pred:.2f}"
    )


st.markdown(f"""
<div style="
background:#1b4332;
padding:20px;
border-radius:15px;
text-align:center;
margin-top:15px;
box-shadow:0px 0px 15px rgba(0,255,0,0.4);
">

<h2 style="color:white;">
🧪{fert_name}
</h2>

<p style="color:#90EE90;font-size:20px;">
Recommended Fertilizer
</p>

</div>
""",unsafe_allow_html=True)



if fert_name=="NPK":

    st.info("""
Nitrogen (N) → Leaf growth

Phosphorus (P) → Root development

Potassium (K) → Disease resistance

Recommended for balanced crop nutrition
""")

elif fert_name=="Urea":

    st.info("""
Urea contains approximately 46% Nitrogen.

It promotes leafy growth and improves plant vigor.

Suitable for crops requiring high nitrogen content.
""")

elif fert_name=="DAP":

    st.info("""
DAP contains Nitrogen and Phosphorus.

It encourages root development and early plant growth.

Commonly used during sowing stage.
""")


st.markdown("---")


fig1=px.scatter(
df,
x="Rainfall",
y="Yield",
color="Crop",
title="Crop Yield Analysis"
)

fig1.update_layout(
template="plotly_dark"
)

st.plotly_chart(
fig1,
use_container_width=True
)



fig2=px.histogram(
df,
x="Crop",
title="Crop Distribution"
)

fig2.update_layout(
template="plotly_dark"
)

st.plotly_chart(
fig2,
use_container_width=True
)



fig3=px.box(
df,
x="Crop",
y="Rainfall",
color="Crop",
title="Rainfall Analysis"
)

fig3.update_layout(
template="plotly_dark"
)

st.plotly_chart(
fig3,
use_container_width=True
)



st.markdown("## 📋 Dataset Preview")

st.dataframe(
df.head(20),
use_container_width=True
)