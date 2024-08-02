from transformers import pipeline
import streamlit as st

def main():
    
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    candidate_idiomas = ['English', 'Español']
    candidate_contextos = {
        'English': ['Politics', 'Cine', 'Religion'],
        'Español': ['Política', 'Cine', 'Religión']
    }

    st.title('Wilson Xavier Monteros Enriquez')

    input_usuario = st.text_input('Inserte su texto a predecir:', '')
    
   

    if st.button('Predecir'):
       
        prediccion_idioma = classifier(input_usuario, candidate_idiomas)
        idioma_label = prediccion_idioma['labels'][0]
        st.write(f'Idioma detectado: {idioma_label}')
        
        prediccion_contexto = classifier(input_usuario, candidate_contextos[idioma_label])
        
        if  idioma_label in candidate_contextos:
            prediccion_contexto = classifier(input_usuario, candidate_contextos[idioma_label])
            contexto_label = prediccion_contexto['labels'][0]
            st.write(f'Predicción de contexto: {contexto_label}')
            
        
           
       
    else:
            st.write("No se pudo detectar un idioma válido para predecir el contexto.")
                 

if __name__ == '__main__':
    main()

