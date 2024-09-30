import streamlit as st
from PIL import Image
from qwen_model import load_model, qwen_ocr_extract, filter_text, highlight_text


def main():
    model, processor = load_model()

    st.title("OCR App")

    uploaded_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="IMAGE", use_column_width=True)

        extracted_text = qwen_ocr_extract(model, processor, image)
        if st.button("Extract text", use_container_width=True):
            with st.spinner("Extracting text...."):
                st.text(extracted_text)

        filtered_text = filter_text(extracted_text)
        if st.button("Filtered text", use_container_width=True):
            with st.spinner("Filtering...."):
                st.text(filtered_text)

        if st.button("Search Keyword", use_container_width=True):
            with st.spinner("Searching...."):
                keyword = st.text_input("Enter Keyword")

                if keyword:
                    search_results = highlight_text(filtered_text, keyword)
                    
                    if search_results:
                        st.subheader("Search Results:")
                        st.write(f"-{search_results}")
                    else:
                        st.write("No results found")
            

        if st.button("Query",use_container_width=True):
            with st.spinner("Thinking...."):
                query = st.text_input("Ask your query")
                if query:
                    
                    query_results = qwen_ocr_extract(model, processor, image, query)

                    if query_results:
                        st.subheader("Query Results:")
                        st.write(f"-{query_results}")
                    else:
                        st.write("No results found")
        
if __name__=="__main__":
    main()
