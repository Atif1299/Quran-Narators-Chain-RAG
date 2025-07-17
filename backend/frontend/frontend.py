import streamlit as st
import requests

# Define base URL
BASE_URL = "http://127.0.0.1:8000"
# Create two columns for the grid layout
col1, col2 = st.columns(2)

# Left column: Hadith Validation using Quran
with col1:
    st.header("Hadith Validation using Quran")
    st.write("Enter a Hadith in English to validate it using similar Quranic ayahs.")
    
    query = st.text_area("Enter your Hadith", height=150, key="hadith_validation_input")
    
    if st.button("Validate Hadith", key="validate_hadith_button"):
        if not query.strip():
            st.warning("Please enter a Hadith.")
        else:
            url = f"{BASE_URL}/api/quran/search"
            payload = {"query": query}
            try:
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    result = data["results"]
    
                    st.success("Hadith validated successfully!")
    
                    st.markdown(f"### Hadith")
                    st.markdown(result["hadith"])
    
                    st.markdown(f"### Verdict")
                    st.markdown(f"**{result['verdict']} ⇒ Confidence: {result['confidence']}**")
    
                    st.markdown(f"### Summary")
                    st.markdown(result["summary"])
    
                    def display_ayahs(ayahs, title):
                        if ayahs:
                            st.markdown(f"### {title}")
                            for ayah in ayahs:
                                st.markdown(f"**Surah:** {ayah['surah_name_english']} - Ayah {ayah['aya_number']}")
                                st.markdown(f"**Score:** {ayah['score']:.4f}")
                                st.markdown(f"**Arabic:** {ayah['arabic_diacritics']}")
                                st.markdown(f"**English Translation:** {ayah['english_translation']}")
                                st.markdown("---")
    
                    display_ayahs(result["supported"], "Supported Ayahs")
                    display_ayahs(result["contradicted"], "Contradicted Ayahs")
    
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")

# Right column: Hadith Narrator Chain Extractor
with col2:
    st.header("Hadith Narrator Chain Extractor")
    st.write("Enter a hadith and select the language. The narrator chain will be extracted and displayed below.")
    
    hadith_text = st.text_area("Hadith Text", height=150, key="hadith_narrator_input")
    language = st.selectbox("Language", ["english", "arabic"], key="language_select")
    
    if st.button("Extract Narrators Chain (Open Source dslim/bert-base-NER)", key="extract_narrators_dslim"):
        if not hadith_text.strip():
            st.warning("Please enter a hadith text.")
        else:
            url = f"{BASE_URL}/api/extraction/extract_narrators_ner_dslim"
            payload = {"hadith_text": hadith_text, "language": "english"}
            try:
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    st.success("Narrators chain extracted successfully!")
                    st.json(data)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
    
    if st.button("Extract Narrators Chain (Open Source CAMeL-Lab/bert-base-arabic-camelbert-msa-ner)", key="extract_narrators_camel"):
        if not hadith_text.strip():
            st.warning("Please enter a hadith text.")
        else:
            url = f"{BASE_URL}/api/extraction/extract_narrators_ner_CAMel_Lab"
            payload = {"hadith_text": hadith_text, "language": "arabic"}
            try:
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    st.success("Narrators chain extracted successfully!")
                    st.json(data)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
    
    if st.button("Extract Narrators Chain (Close Source)", key="extract_narrators_llm"):
        if not hadith_text.strip():
            st.warning("Please enter a hadith text.")
        else:
            url = f"{BASE_URL}/api/extraction/extract_narrators_llm"
            payload = {"hadith_text": hadith_text, "language": language}
            try:
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    st.success("Narrators chain extracted successfully!")
                    st.json(data)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")