import streamlit as st
from dotenv import find_dotenv, load_dotenv
import os
from gtts import gTTS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline

# To be used when running locally
# Also .env file to be created at root folder level
# with token: HUGGINGFACEHUB_API_TOKEN = <Your HuggingFace Hub API Token>
# load_dotenv(find_dotenv())
# To be used when deploying to Streamlit Cloud
hf_token = st.secrets["HUGGINGFACE_TOKEN"]["token"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# Cache the model loading to improve performance
@st.cache_resource
def load_generator():
    """Cache the model loading to avoid reloading on every rerun"""
    return pipeline(
        "text-generation",
        model="distilgpt2",
        max_length=100
    )

# Generate Story using Langchain
def story_generator(scenario):
    template = """
    Write a very short story based on this scene: {scenario}
    The story should be creative and engaging but no more than 60 words.
    """

    try:
        # Get the cached model
        text_generator = load_generator()
        
        # Wrap the pipeline in LangChain
        llm = HuggingFacePipeline(pipeline=text_generator)
        
        # Create the prompt template
        prompt = PromptTemplate(
            template=template,
            input_variables=["scenario"]
        )
        
        # Create the chain
        story_llm = LLMChain(prompt=prompt, llm=llm)
        
        # Generate the story
        with st.spinner('Generating story...'):
            story = story_llm.predict(scenario=scenario)
        
        # Clean up the output
        story = story.strip()
        if not story:
            raise ValueError("Generated story is empty")
            
        return story
        
    except Exception as e:
        st.error(f"Error in story generation: {str(e)}")
        return f"Story generation failed: {str(e)}"
        
# Text to Audio
def text_to_audio(story):
    try:
        with st.spinner('Converting to audio...'):
            language = 'en'
            audio = gTTS(text=story, lang=language, slow=False)
            audio.save("audio.mp3")
    except Exception as e:
        st.error(f"Error in audio generation: {str(e)}")

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
    st.markdown("The code for this App is available on [GitHub](https://github.com/ajtechdeveloper/AIPictureToAudioStory)")
    st.markdown("For a full tutorial about this App, please refer to my blog post: [Generative AI App LangChain Hugging Face Open Source Models Tutorial](https://softwaredevelopercentral.blogspot.com/2024/05/generative-ai-app-langchain-hugging.html)")
    # Add a loading message while the model is being loaded
    if 'model_loaded' not in st.session_state:
        with st.spinner('Loading AI model... This may take a minute on first run...'):
            load_generator()
            st.session_state.model_loaded = True
    
    uploaded_file = st.file_uploader("Choose an image", type="jpg")

    if uploaded_file is not None:
        try:
            # Create a placeholder for the image
            st.image(uploaded_file, caption="Uploaded Image", width=600)
            
            # Save the file temporarily
            bytes_data = uploaded_file.getvalue()
            with open(uploaded_file.name, "wb") as file:
                file.write(bytes_data)
            
            # Generate caption
            with st.spinner('Analyzing image...'):
                scenario = image_to_text(uploaded_file.name)
            
            # Generate story
            story = story_generator(scenario)
            
            if not story.startswith("Story generation failed"):
                # Generate audio
                text_to_audio(story)
                
                # Display results in a clean layout
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("📝 Caption", expanded=True):
                        st.write(scenario)
                with col2:
                    with st.expander("📖 Story", expanded=True):
                        st.write(story)
                
                # Audio player
                st.subheader("🎧 Listen to the Story")
                st.audio("audio.mp3")
            else:
                st.error(story)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            # Clean up the temporary file
            if os.path.exists(uploaded_file.name):
                os.remove(uploaded_file.name)

if __name__ == '__main__':
    main()
