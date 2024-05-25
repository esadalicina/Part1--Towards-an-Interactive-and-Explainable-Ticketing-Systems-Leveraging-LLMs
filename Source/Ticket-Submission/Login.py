
import streamlit as st

def main():
    st.title("Creative Login Website")

    st.sidebar.title("Login")
    email = st.sidebar.text_input("Email")
    name = st.sidebar.text_input("Name")
    password = st.sidebar.text_input("Password", type="password")
    login_button = st.sidebar.button("Login")

    if login_button:
        if email and name and password:
            # Here, you can add your authentication logic
            st.success(f"Welcome, {name}!")
        else:
            st.error("Please fill in all fields.")

if __name__ == "__main__":
    main()