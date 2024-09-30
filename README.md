#  Optical Character Recognition and Extraction app with ColPali implementation of the new Byaldi library + Huggingface transformers for Qwen2-VL.
[![Open the App in Streamlit] (link = https://ocr-app-rohit-singh.streamlit.app)]


### How to run it on your own machine
This README provides a step-by-step guide on how to run OCR App using Colpali Byaldi and Qwen2-VL models in your device locally.

### First most step clone the github repo with this link below
```bash
  git clone https://github.com/Rohit-515/OCR_app_ColPali_Qwen2VL.git
```

### **Installation to run app locally**
1. **Create a Python environment (optional):**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```
2. **Install required libraries:**
   In terminal write this code to install the libraries
   ```bash
   pip install -r requirements
   ```
3. **Run the app using following command in terminal**

   ```
   $ streamlit run streamlit_app.py
   
   //use this command if getting Axios403 error
   $ streamlit run app.py --server.enableXsrfProtection false


   ```
