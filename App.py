import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Delta Test", layout="wide")
st.title("Delta API Test")

st.write("Testing API connection...")

try:
    r = requests.get(
        "https://api.india.delta.exchange/v2/products",
        params={"contract_types": "perpetual_futures"},
        timeout=15
    )
    st.write("Status Code:", r.status_code)
    
    if r.status_code == 200:
        data = r.json()
        st.write("Success:", data.get("success"))
        result = data.get("result", [])
        st.write("Total products:", len(result))
        
        if result:
            st.success("API working!")
            # Show first 5 symbols
            symbols = [item.get("symbol") for item in result[:10]]
            st.write("Sample coins:", symbols)
        else:
            st.error("No products returned")
    else:
        st.error(f"API Error: {r.status_code}")
        st.write(r.text[:500])

except Exception as e:
    st.error("Exception aaya:")
    st.write(str(e))
