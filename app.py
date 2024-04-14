import streamlit as st
from dotenv import find_dotenv, load_dotenv
import os
from gtts import gTTS
from langchain.chains.llm import LLMChain
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from transformers import pipeline

# To be used when running locally
# Also .env file to be created at root folder level
# with token: HUGGINGFACEHUB_API_TOKEN = <Your HuggingFace Hub API Token>
# load_dotenv(find_dotenv())
# To be used when deploying to Streamlit Cloud
hf_token = st.secrets["HUGGINGFACE_TOKEN"]["token"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token


# Text to Audio
def text_to_audio(story):
    # Language to be used
    language = 'en'
    # Create Audio
    audio = gTTS(text=story, lang=language, slow=False)
    # Save the audio converted as a mp3 file
    audio.save("audio.mp3")


# Generate Story using Langchain
def story_generator(scenario):
    template = """
    You are a story teller;
    You can generate short stories based on a simple narrative
    Your story should be no more than 60 words.

    CONTEXT: {scenario}
    STORY:
    """

    repo_id = "HuggingFaceH4/zephyr-7b-beta"
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 1, "max_length": 64})
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_llm = LLMChain(prompt=prompt, llm=llm)
    story = story_llm.predict(scenario=scenario)
    spl_word = 'STORY:'
    res = story.split(spl_word, 1)
    actual_story = res[1]
    print(actual_story)
    return actual_story


# Image to text
def image_to_text(url):
    caption_creator = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = caption_creator(url)[0]["generated_text"]
    print(text)
    return text


def main():
    st.set_page_config(page_title="Picture to Audio Story", page_icon=":)")
    st.header("AI In Action: Transform A Picture To An Audio Story")
    st.markdown("This App uses AI to generate a caption for any uploaded picture and a short audio story using the caption.")
    uploaded_file = st.file_uploader("Choose an image", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        scenario = image_to_text(uploaded_file.name)
        story = story_generator(scenario)
        text_to_audio(story)

        with st.expander("Caption"):
            st.write(scenario)
        with st.expander("Story"):
            st.write(story)
        st.audio("audio.mp3")


if __name__ == '__main__':
    main()
