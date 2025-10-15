import streamlit as st
from io import BytesIO
from PIL import Image, ImageDraw

st.title('String to Image App')

text = st.text_input('Enter some text:')

if st.button('Generate image'):
    if not text.strip():
        st.error('Please enter a non-empty string.')
    else:
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((20, 40), text, fill='black')

        st.image(img, caption='Generated image')

        buf = BytesIO()
        img.save(buf, format='PNG')
        st.download_button(
            label='Download image',
            data=buf.getvalue(),
            file_name='text_image.png',
            mime='image/png'
        )
