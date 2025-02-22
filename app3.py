from flask import Flask, render_template, request ,jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai

app = Flask(__name__)
load_dotenv()

class AITextDetector:
    def __init__(self):
        manual_api_key = "AIzaSyDVCdu-XGZRa6_ty_Fc32RCdz1Nb4zy0gk"  # Replace with your real key
        genai.configure(api_key=manual_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.safety_settings = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
        }

    def detect(self, text):
        try:
            if len(text) < 100:
                return {"error": "Text must be at least 100 characters"}

            prompt = f"""Analyze this text and determine if it's AI-generated. Consider:
            1. Sentence structure complexity
            2. Creativity/originality of ideas
            3. Presence of common AI patterns
            4. Consistency of writing style
            5. Contextual understanding depth

            Return response as JSON format only:
            {{
                "verdict": "AI-generated" or "Human-written",
                "confidence": 0-100,
                "reason": "short explanation"
            }}

            Text to analyze: {text}"""

            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings
            )

            
            try:
                analysis = eval(response.text.replace('```json', '').replace('```', '').strip())
                return analysis
            except:
                return {"error": "Failed to parse Gemini response"}

        except Exception as e:
            return {"error": f"Gemini API Error: {str(e)}"}

@app.route('/cover_letter')
def cover_letter_form():
    return render_template('cover_letter.html')

@app.route('/generate_cover_letter', methods=['POST'])
def generate_cover_letter():
    if request.method == 'POST':
        form_data = {
            'full_name': request.form['full_name'],
            'email': request.form['email'],
            'company_name': request.form['company_name'],
            'hiring_manager': request.form['hiring_manager'],
            'position': request.form['position'],
            'content': request.form['content']
        }

        detector = AITextDetector()
        result = detector.detect(form_data['content'])

        if 'error' in result:
            return render_template('error.html', message=result['error'])

        return render_template('cover_letter_analysis.html',
                            analysis=result,
                            **form_data)

if __name__ == '__main__':
    app.run(debug=True)