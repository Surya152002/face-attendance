import streamlit as st
import pandas as pd
import cv2
import base64
import os
import numpy as np
import datetime
import csv

# Your existing Python code for face recognition, etc.
# ...

# Function to check login credentials (Dummy function, replace with real logic)
def check_login(username, password):
    return username == "admin" and password == "admin"

# Function to encode a file for download
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="attendance.csv">Download CSV</a>'
    return href

def main():
    st.title("Face Recognition and Attendance System")

    # Login System
    if 'login_status' not in st.session_state:
        st.session_state['login_status'] = False

    if st.session_state['login_status']:
        # Main Interface
        uploaded_file = st.file_uploader("Upload an image for face recognition", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            if st.button("Recognize Faces"):
                # Recognize faces from the uploaded image
                bboxs, _ = get_face(opencv_image)
                for i, (x1, y1, x2, y2) in enumerate(bboxs):
                    face_image = opencv_image[y1:y2, x1:x2]
                    recognition(face_image, i)
                st.image(opencv_image, channels="BGR", caption="Processed Image")

        if 'recognized_names' in globals():
            # Show Attendance Table
            if recognized_names:
                st.table(pd.DataFrame(recognized_names, columns=["Name"]))
                # Download link for CSV
                st.markdown(get_table_download_link(pd.DataFrame(recognized_names, columns=["Name"])), unsafe_allow_html=True)

    else:
        # Login Page
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if check_login(username, password):
                st.session_state['login_status'] = True
            else:
                st.error("Incorrect Username/Password")

if __name__ == "__main__":
    main()

