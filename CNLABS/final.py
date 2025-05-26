# import wikipedia as wiki
# import wikipedia as wiki
# import numpy as np
# import tensorflow as tf
# from tensorflow.data import Dataset as tf_dataset
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,LSTM,Embedding
# from tensorflow.keras.models import load_model
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'



# def generate_text(start_string, gen_length=300, temperature=0.1):
#     input_eval = [char_index.get(s, 0) for s in start_string.lower()]

#     if len(input_eval) < seq_length:
#         input_eval = [0] * (seq_length - len(input_eval)) + input_eval
#     else:
#         input_eval = input_eval[-seq_length:]

#     input_eval = tf.expand_dims(input_eval, 0)
#     text_generated = []

#     for _ in range(gen_length):
#         predictions = model(input_eval)
#         predictions = tf.squeeze(predictions, 0)[-1].numpy()

#         predictions = predictions / temperature
#         predicted_id = np.random.choice(len(predictions), p=tf.nn.softmax(predictions).numpy())

#         input_eval = tf.concat([input_eval[:, 1:], tf.expand_dims([predicted_id], 1)], axis=1)
#         text_generated.append(str(index_char[predicted_id]))

#     return start_string + ''.join(text_generated)

# def search_data(topics):
#   Search = ""
#   try:
#     for searc in topics:
#       Search = wiki.search(searc)
#       print("\nResults Found:",Search)
#     print("Data search is succesfully")
#   except Exception as e:
#     print("Error in search result:",e)
#   return Search

#   def get_data(topics):
#     data = ''
#     try:
#       for Data in topics:
#         pages = wiki.page(Data)
#         data+=pages.content.lower()
#     except wiki.exceptions.PageError:
#       print('Page Error:',{Data})

#     except wiki.exceptions.DisambiguationError as e:
#       print('Disambiguation Error:',{Data})

#     except Exception as e:
#       print("Error in Extracting the Data:",e)
#     return data
  
#   def preprocess_data(data):
#     vocab = sorted(set(data))
#     char_index = {c:i for i,c in enumerate(vocab)}
#     index_char = np.array(vocab)
#     txt_as_int = np.array([char_index[c] for c in data])
#     return txt_as_int,vocab,char_index,index_char
# topics = ['Artificial Intelligence']
# s_result = search_data(topics)
# print(s_result)


# data = get_data(topics)
# print("Dataset:",data)
# print("Length of the Dataset:",len(data))

# pre_data,vocab,char_index,index_char = preprocess_data(data)
# print(pre_data)


# def seq_data(text_as_int):
#   seq = 100
#   char_dataset = tf_dataset.from_tensor_slices(text_as_int)
#   sequences = char_dataset.batch(seq+1,drop_remainder=True)
#   return sequences


# def split_input(chunk):
#   inp = chunk[:-1]
#   out = chunk[1:]
#   return inp,out


# sequences = seq_data(pre_data)
# Seq = sequences.map(split_input)

# def batch_data(Seq):
#   buffer_size = 1000
#   BATCH_SIZE = 128
#   return Seq.shuffle(buffer_size).batch(BATCH_SIZE,drop_remainder=True)



# model = Sequential([
#     Embedding(len(vocab),256,input_length=100),
#     LSTM(512,return_sequences=True),
#     Dense(len(vocab))
# ])
# model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# model= load_model("epoch_200.keras",compile=False)

# print(generate_text("Artificial Intelligence", gen_length=400, temperature=0.1))



import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding
import numpy as np
import wikipedia as wiki
from tensorflow.data import Dataset as tf_dataset
import os

# Environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

seq_length=100
# Text Generation Function
def generate_text(start_string, gen_length=300, temperature=0.1):
    input_eval = [char_index.get(s, 0) for s in start_string.lower()]

    if len(input_eval) < seq_length:
        input_eval = [0] * (seq_length - len(input_eval)) + input_eval
    else:
        input_eval = input_eval[-seq_length:]

    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    for _ in range(gen_length):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)[-1].numpy()

        predictions = predictions / temperature
        predicted_id = np.random.choice(len(predictions), p=tf.nn.softmax(predictions).numpy())

        input_eval = tf.concat([input_eval[:, 1:], tf.expand_dims([predicted_id], 1)], axis=1)
        text_generated.append(str(index_char[predicted_id]))

    return start_string + ''.join(text_generated)


# Search Wikipedia Topics
def search_data(topics):
    search = ""
    try:
        for searc in topics:
            search = wiki.search(searc)
            print("\nResults Found:", search)
        print("Data search is successful")
    except Exception as e:
        print("Error in search result:", e)
    return search


# Fetch Wikipedia Data
def get_data(topics):
    data = ''
    try:
        for topic in topics:
            pages = wiki.page(topic)
            data += pages.content.lower()
    except wiki.exceptions.PageError:
        print('Page Error:', {topic})
    except wiki.exceptions.DisambiguationError as e:
        print('Disambiguation Error:', {topic})
    except Exception as e:
        print("Error in Extracting the Data:", e)
    return data


# Preprocess Data
def preprocess_data(data):
    vocab = sorted(set(data))
    char_index = {c: i for i, c in enumerate(vocab)}
    index_char = np.array(vocab)
    txt_as_int = np.array([char_index[c] for c in data])
    return txt_as_int, vocab, char_index, index_char


# Create Sequence Data
def seq_data(text_as_int):
    char_dataset = tf_dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
    return sequences


# Split Input and Target
def split_input(chunk):
    inp = chunk[:-1]
    out = chunk[1:]
    return inp, out


# Batch Data
def batch_data(seq):
    buffer_size = 1000
    BATCH_SIZE = 128
    return seq.shuffle(buffer_size).batch(BATCH_SIZE, drop_remainder=True)


# Main logic
topics = ['Artificial Intelligence']
s_result = search_data(topics)
print(s_result)

data = get_data(topics)
print("Dataset:", data)
print("Length of the Dataset:", len(data))

pre_data, vocab, char_index, index_char = preprocess_data(data)
print(pre_data)

seq_length = 100
sequences = seq_data(pre_data)
Seq = sequences.map(split_input)
batched_dataset = batch_data(Seq)

# Load model
model = load_model("epoch_200.keras", compile=False)

# Example generation at startup
#print(generate_text("Artificial Intelligence", gen_length=400, temperature=0.1))

# Streamlit UI
st.set_page_config(page_title='Basic LLM Chatbot')
st.title("LLM Chatbot")
st.info('The LLM is trained on Artificial Intelligence Wikipedia Data')
st.sidebar.selectbox("Generation Length",options=[1000,3000,4000])
st.sidebar.selectbox("Temperature:",options=[0.1,0.2,0.3])

prompt = st.chat_input('Enter Your Prompt')

if prompt:
    if 'chat' not in st.session_state:
        st.session_state.chat = []

    st.chat_message('user').write(prompt)

    with st.spinner("Thinking.."):
        response = generate_text(prompt, gen_length=400, temperature=0.1)

    st.session_state.chat.append({'role': 'assistant', 'content': response})

    with st.chat_message('assistant'):
        st.write(response)
