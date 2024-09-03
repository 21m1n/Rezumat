import gradio as gr
import PyPDF2
import io

def process_input(text_input, additional_text, file_upload, input_type, api_key, interface, model, technical_skills, soft_skills, experience, education):
    content = ""
    
    # Process text input
    if text_input:
        content += f"## Main Text Input\n\n{text_input}\n\n"
    
    # Process additional text or file upload
    if input_type == "Text" and additional_text:
        content += f"## Additional Text Input\n\n{additional_text}\n\n"
    elif input_type == "File" and file_upload is not None:
        try:
            if file_upload.name.endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_upload.read()))
                file_content = ""
                for page in pdf_reader.pages:
                    file_content += page.extract_text() + "\n"
            else:
                file_content = file_upload.read().decode('utf-8')
            content += f"## Uploaded File Content\n\n{file_content}\n\n"
        except Exception as e:
            content += f"Error processing file: {str(e)}\n\n"
    
    # Add processing details
    content += f"API Key: {'*' * len(api_key)}\nInterface: {interface}\nModel: {model}\n"
    content += f"Weightage:\n- Technical Skills: {technical_skills}%\n- Soft Skills: {soft_skills}%\n- Experience: {experience}%\n- Education: {education}%"
    
    return content

def reset_interface():
    return (
        "", "Text", "", None, "", 60, 10, 10, 10
    )

with gr.Blocks() as demo:
    gr.Markdown("# ✏️ Resume Evaluator")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Weightage")
            api_key22 = gr.Textbox(label="API Key", type="password")
            technical_skills = gr.Slider(minimum=0, maximum=100, value=60, step=1, label="Technical Skills")
            soft_skills = gr.Slider(minimum=0, maximum=100, value=10, step=1, label="Soft Skills")
            experience = gr.Slider(minimum=0, maximum=100, value=10, step=1, label="Experience")
            education = gr.Slider(minimum=0, maximum=100, value=10, step=1, label="Education")
            # Side column (20% of the screen)
            api_key = gr.Textbox(label="API Key", type="password")
            interface = gr.Dropdown(["Groq", "OpenAI", "Anthropic"], label="Interface", value="Groq")
            model = gr.Dropdown(["llama3-70b-8192", "gpt-3.5-turbo", "gpt-4"], label="Model", value="llama3-70b-8192")
            
            # gr.Markdown("### Weightage")

        
        with gr.Column(scale=8):
            # Main content (80% of the screen)
            text_input = gr.Textbox(label="Enter your text", lines=15)
            
            input_type = gr.Radio(["Text", "File"], label="Input Type", value="Text")
            
            with gr.Row() as additional_input_row:
                additional_text = gr.Textbox(label="Additional Text Input", lines=5, visible=False)
                file_upload = gr.File(label="Upload File", file_count="multiple", file_types=["pdf"], visible=False)
            
            output = gr.Markdown(label="Output")
    
            with gr.Row():
                submit_btn = gr.Button("Submit")
                reset_btn = gr.Button("Reset")
    
    def update_input_type(choice):
        if choice == "Text":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)
    
    input_type.change(update_input_type, inputs=[input_type], outputs=[additional_text, file_upload])
    
    submit_btn.click(
        fn=process_input,
        inputs=[text_input, additional_text, file_upload, input_type, api_key, interface, model, technical_skills, soft_skills, experience, education],
        outputs=output
    )
    
    reset_btn.click(
        fn=reset_interface,
        inputs=[],
        outputs=[text_input, input_type, additional_text, file_upload, output, technical_skills, soft_skills, experience, education]
    )

demo.launch()