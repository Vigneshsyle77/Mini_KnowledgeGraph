import streamlit as st
import hashlib

# Hardcoded users (Later replace with DB)
USERS = {
    "admin": "21232f297a57a5a743894a0e4a801fc3",  # password: admin
    "test": "098f6bcd4621d373cade4e832627b4f6",   # password: test
}

def hash_pass(password):
    return hashlib.md5(password.encode()).hexdigest()

def login():
    st.title("🔐 AI-KnowMap Login")
    st.write("Enter your credentials to continue.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == hash_pass(password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success("Login successful 🎉")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password ❌")
