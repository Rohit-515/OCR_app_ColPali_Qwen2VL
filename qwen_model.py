import streamlit as st
import torch
import re
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from byaldi import RAGMultiModalModel

RAG = RAGMultiModalModel.from_pretrained("vidore/colpali", verbose=0)

@st.cache_resource
def load_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            ).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",trust_remote_code=True)

    return model, processor


# Indexing
def rag_index(image):
    RAG.index(
        input_path=image,
        index_name="image_index",
        store_collection_with_index=False,
        overwrite=True,
    )

# extraction and quering
def qwen_ocr_extract(model, processor, image, query):
    try:
        rag_index(image)
        
        if query:
            text_query = query
        else:
            text_query = "Extract the text"

        results = RAG.search(text_query, k=1)
        image_index = results[0]
        img = Image.open(image_index)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img,
                    },
                    { "type": "text", "text": text_query},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs = process_vision_info(messages)
        inputs = processor(
            text = [text],
            image = image_inputs,
            padding = True,
            return_tensors="pt",
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = inputs.to(device)

        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids): ] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokena=True, clean_up_tokenization_spaces=False
        )

        return output_text
    
    except Exception as e:
        return f"Error:{e}"



# text filter
def filter_text(output_text):
    try:
        text = output_text[0]
        lines = text.split('\n')

        filtered_text = []
        for line in lines:
            if re.match(r'[A-Za-z0-9]', line):  
                filtered_text.append(line.strip())

        return "\n".join(filtered_text)
    
    except Exception as e:
        return f"Error:{e}"


# highlighter
def highlight_text(filtered_text, keyword):
    try:
        highlighted_text = filtered_text

        if keyword=='':
            return "Please enter a keyword"
        
        else:
            for word in keyword.split():
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                highlighted_text = pattern.sub(lambda m: f'<span style="background-color: blue;">{m.group()}</span>', highlighted_text)

        return highlighted_text
    
    except Exception as e:
        return f"Error:{e}"
