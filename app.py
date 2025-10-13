import streamlit as st
import os
from transformers import pipeline
from gtts import gTTS
from huggingface_hub import login

# To be used when running locally
# Also .env file to be created at root folder level
# with token: HUGGINGFACEHUB_API_TOKEN = <Your HuggingFace Hub API Token>
# load_dotenv(find_dotenv())
# To be used when deploying to Streamlit Cloud
hf_token = st.secrets["HUGGINGFACE_TOKEN"]["token"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
login(token=hf_token)  # Login to Hugging Face

def image_to_text(image_path):
    """Generate caption for the uploaded image"""
    try:
        # Initialize the image captioning model
        caption_model = pipeline(
            "image-to-text", 
            model="Salesforce/blip-image-captioning-base",
            token=hf_token
        )
        
        # Generate caption
        result = caption_model(image_path)
        
        # Return the generated caption
        return result[0]['generated_text']
    except Exception as e:
        st.error(f"Error in image captioning: {str(e)}")
        raise

def story_generator(scenario):
    """Generate a short story based on the image caption"""
    try:
        # Initialize the text generation model
        generator = pipeline(
            'text2text-generation',
            model='google/flan-t5-large',
            token=hf_token
        )

        # Template with context
        template = """
        You are a story teller;
        You can generate short stories based on a simple narrative
        Your story should be no more than 60 words.
        
        CONTEXT: {scenario}
        STORY:"""
        
        # Format the template with the scenario
        context = template.format(scenario=scenario)
        
        # Generate story
        result = generator(
            context,
            max_length=100,
            num_return_sequences=1,
            temperature=1,
            do_sample=True
        )
        
        # Extract and clean the story
        story = result[0]['generated_text'].strip()
        # If story is too short, try again
        if len(story.split()) < 10:
            return story_generator(scenario)

        return story
    except Exception as e:
        st.error(f"Error in story generation: {str(e)}")
        raise

def text_to_audio(text):
    """Convert text to audio"""
    try:
        # Create audio file
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("audio.mp3")
    except Exception as e:
        st.error(f"Error in audio generation: {str(e)}")
        raise

def main():
    # Set up the Streamlit page
    st.set_page_config(
        page_title="AI Picture to Audio Story",
        page_icon="📸"
    )
    
    st.title("AI In Action: Transform A Picture To An Audio Story")
    st.markdown("This App uses AI to generate a caption for any uploaded picture and a short audio story using the caption.")
    st.markdown("The code for this App is available on [GitHub](https://github.com/ajtechdeveloper/AIPictureToAudioStory)")
    st.markdown("For a full tutorial about this App, please refer to my blog post: [Generative AI App LangChain Hugging Face Open Source Models Tutorial](https://softwaredevelopercentral.blogspot.com/2024/05/generative-ai-app-langchain-hugging.html)")
    
    
    # Add information about the app
    st.markdown("""
    This app:
    1. Generates a caption for your image 📝
    2. Creates a short story based on the caption 📖
    3. Converts the story to audio 🎧
    """)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Create a spinner for image processing
            with st.spinner('Processing your image...'):
                # Display the uploaded image
                st.image(uploaded_file, caption='Uploaded Image', width=400)
                
                # Save the uploaded file temporarily
                with open("temp_image.jpg", "wb") as f:
                    f.write(uploaded_file.getvalue())

            # Generate image caption
            with st.spinner('Analyzing image and generating caption...'):
                caption = image_to_text("temp_image.jpg")
                st.subheader("📝 Image Caption")
                st.write(caption)

            # Generate story
            with st.spinner('Creating a story based on the caption...'):
                story = story_generator(caption)
                st.subheader("📖 Generated Story")
                st.write(story)

            # Create audio
            with st.spinner('Converting story to audio...'):
                text_to_audio(story)
                st.subheader("🎧 Listen to the Story")
                st.audio("audio.mp3")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
        finally:
            # Clean up temporary files
            if os.path.exists("temp_image.jpg"):
                os.remove("temp_image.jpg")

if __name__ == "__main__":
    main()
