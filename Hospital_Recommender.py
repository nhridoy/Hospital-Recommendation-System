# importing libraries
import folium
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
from geopy.geocoders import MapBox
from geopy import distance
from streamlit_folium import folium_static

# loading hospital data
hos = "Hospital Lists.xlsx"
hospital = pd.read_excel(hos, sheet_name=0)

hospital = hospital.iloc[:, :-1]
hospital = hospital.iloc[:, :-1]

# loading cardio data in a variable
csv = "cardio.csv"
# opening cardio data
df = pd.read_csv(csv)

# Converting Days into Years
df["years"] = (df["age"] / 365).round(0)
df["years"] = pd.to_numeric(df["years"], downcast="integer")

# Deleting ID and age column
df = df.drop("id", axis=1)
df = df.drop("age", axis=1)

# Fixing Outliners years, height, weight, ap_hi and ap_lo
s_list = ["years", "height", "weight", "ap_hi", "ap_lo"]
def standartization(df):
    x_std = df.copy(deep=True)
    for column in s_list:
        x_std[column] = (x_std[column]-x_std[column].mean())/x_std[column].std()
    return x_std 
x_std=standartization(df)
x_melted = pd.melt(frame=x_std, id_vars="cardio", value_vars=s_list, var_name="features", value_name="value", col_level=None)
ap_list = ["ap_hi", "ap_lo"]
boundary = pd.DataFrame(index=["lower_bound","upper_bound"]) # We created an empty dataframe
for each in ap_list:
    Q1 = df[each].quantile(0.25)
    Q3 = df[each].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1- 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    boundary[each] = [lower_bound, upper_bound ]
ap_hi_filter = (df["ap_hi"] > boundary["ap_hi"][1])
ap_lo_filter = (df["ap_lo"] > boundary["ap_lo"][1])                                                           
outlier_filter = (ap_hi_filter | ap_lo_filter)
x_outliers = df[outlier_filter]
out_filter = ((df["ap_hi"]>250) | (df["ap_lo"]>200) )
df = df[~out_filter]
# Fixed Outliners

# Creating Title and Subtitle
st.title(
    """
    Hospital Recommendation System
    Recommending Hospital Using AI
    """
)

# Opening and Displaying an Image
# im = requests.get("F:\My Coding\Python\im.jpeg")
# image = Image.open(BytesIO(im.content))
image = Image.open("im.jpeg")
st.image(image, caption="Machine Learning", use_column_width=True)

# Creating Subheader
st.subheader("Data Information:")
# Showing the Data
st.dataframe(df.head(300))
# Showing Statistics of Data
st.write(df.describe())
# Showing Data as a chart
chart = st.bar_chart(df.head(300))

# Splitting Data into feature data X and target data Y Variable
X = df.drop("cardio", axis=1).values
Y = df["cardio"].values

# Splitting Data into 99% training and 1% testing dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=1)


# Getting Input from User
def get_user_input():
    gen = ("female", "male")
    gen_options = list(range(len(gen)))
    gender = st.sidebar.selectbox("gender", gen_options, format_func=lambda x: gen[x])
    height = st.sidebar.slider("height", 30.0, 350.0, 175.26)
    st.sidebar.write(height)
    weight = st.sidebar.slider("weight", 20.0, 180.0, 81.0)
    st.sidebar.write(weight)
    bp_upper = st.sidebar.text_input("bp_upper", 104)
    bp_lower = st.sidebar.text_input("bp_lower", 78)
    col = ("Normal", "Above Normal", "Well Above Normal")
    col_options = list(range(len(col)))
    cholesterol = st.sidebar.selectbox("cholesterol", col_options, format_func=lambda x: col[x])
    gluc = ("Normal", "Above Normal", "Well Above Normal")
    gluc_options = list(range(len(gluc)))
    glucose = st.sidebar.selectbox("glucose", gluc_options, format_func=lambda x: gluc[x])
    smoke = ("No", "Yes")
    smoke_options = list(range(len(smoke)))
    smoker = st.sidebar.selectbox("smoker", smoke_options, format_func=lambda x: smoke[x])
    alc = ("No", "Yes")
    alc_options = list(range(len(alc)))
    alcoholic = st.sidebar.selectbox("alcoholic", alc_options, format_func=lambda x: alc[x])
    actv = ("Not Active", "Active")
    actv_options = list(range(len(actv)))
    activeness = st.sidebar.selectbox("activeness", actv_options, format_func=lambda x: actv[x])
    age = st.sidebar.text_input("age", 24)

    # User Info Dictionary
    user_data = {
        "gender": gender + 1,
        "height": height,
        "weight": weight,
        "ap_high": bp_upper,
        "ap_lo": bp_lower,
        "cholesterol": cholesterol + 1,
        "gluc": glucose + 1,
        "smoke": smoker,
        "alco": alcoholic,
        "active": activeness,
        "years": age
    }
    # Converting User Data into DataFrame
    user_features = pd.DataFrame(user_data, index=[0])

    return user_features


# Storing User Input to a variable
user_input = get_user_input()

address = st.sidebar.text_input("address (Road Name, Village, District, Country)",
                                "New Jail Road, Mohiskhola, Narail, Bangladesh")

# Creating a subheader and displaying the user input
st.subheader("User Input")
st.write(user_input)

# Create and Train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Showing the model accuracy
st.subheader("Model Test Accuracy Score: ")
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100), "%")

# Store The Model Prediction in a Variable
prediction = RandomForestClassifier.predict(user_input)

# Set a subheader and display the classification
st.subheader("Classification")
if prediction == 0:
    st.write("You Don't Have a Cardio Vascular Disease")
else:
    st.write("You Might Have Cardio Vascular Disease")
st.write(prediction)

go = MapBox(api_key="pk.eyJ1IjoiaHJpZG95Ym9zczEyIiwiYSI6ImNrbGo5OW9pbzBnNDgyb28wdG0ycDU1MmQifQ.NxMoHVOobdijNONLuY8QMQ")
add = go.geocode(address)

user_address = f"{add.latitude}, {add.longitude}"

ls = []
for i in range(hospital["Co-Ordinates"].count()):
    ds = distance.distance(user_address, hospital["Co-Ordinates"][i]).km
    ls.append(ds)
hospital["Distance"] = ls
hospital = hospital.sort_values(by="Distance").head(5)

st.subheader("Your Location: ")
st.write(add, add.latitude, add.longitude)
st.subheader("Closest 5 Hospitals from your Location: ")
# Resetting Hospital Index
hospital = hospital.reset_index(drop=True)
st.write(hospital)

# splitting co-ordinates into lattitude and longitude column
hospital[["lat", "lon"]] = hospital['Co-Ordinates'].str.split(',', expand=True)

# Creating Folium Map
m = folium.Map(location=[add.latitude, add.longitude])

# Creating Users Circle Marker and Hospitals Markers
for i in range(hospital["lat"].count()):
    n = hospital["Hospital Name"][i]
    a = hospital["Address"][i]
    d = hospital["District"][i]
    lt = hospital["lat"][i]
    ln = hospital["lon"][i]
    tooltip = f"<i><b>{n}</b></i> in <i><b>{a}</b></i>, {d}"
    folium.Marker(
        [lt, ln], popup=tooltip, tooltip=n
    ).add_to(m)
    folium.CircleMarker(location=[add.latitude, add.longitude], radius=6, popup=f"{add}", tooltip="Your Location",
                        fill_color="blue", color="white", fill_opacity=0.7).add_to(m)

# Setting Dynamic Zoom Level For the Map
sw = hospital[['lat', 'lon']].min().values.tolist()
ne = hospital[['lat', 'lon']].max().values.tolist()

m.fit_bounds([sw, ne])

# Showing The Map
folium_static(m)

# Old Map For Reference
map_data2 = hospital.iloc[:, -2:]
map_data2['lat'] = map_data2['lat'].astype(float)
map_data2['lon'] = map_data2['lon'].astype(float)

st.map(map_data2)
