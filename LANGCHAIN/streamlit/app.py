import streamlit as st
import pandas as pd
import numpy as np
st.write("Hellow world")
hobby=st.text_input("what is your hpbby?")
st.write(f"my Hobby is : {hobby}")

st.button("Click Me")
st.balloons()
st.caption("Hello")

data = pd.read_csv("movies.csv")
st.write(data)
chart_data = pd. DataFrame(
np. random. randn(20, 3), columns=["a", "b", "c"])
st. bar_chart (chart_data) 
st. line_chart (chart_data)

