from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from gradio_client import Client, handle_file
from dotenv import load_dotenv
load_dotenv()

import tempfile
import os
import shutil
import logging

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.environ.get("HF_TOKEN")
client = Client("multimodalart/flux-fill-outpaint", hf_token=HF_TOKEN)

os.makedirs('templates', exist_ok=True)
template_path = os.path.join('templates', 'index.html')

if not os.path.exists(template_path):
    with open(template_path, 'w') as f:
        f.write('''
        <!DOCTYPE html>
        <html>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
        <head>
            <title>Image Outpainting Tool</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                .form-container { margin-top: 20px; }
                #loading { display: none; margin-top: 20px; }
                #result { margin-top: 20px; }
                img { max-width: 100%; }
            </style>
        </head>
        <body>
            <h1>Image Outpainting Tool</h1>
            <div class="form-container">
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" id="imageInput" name="image" accept="image/*" required>
                    <button type="submit">Process Image</button>
                </form>
            </div>
            <div id="loading">Processing... This may take a minute or two.</div>
            <div id="result"></div>

            <script>
                document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    
                    const formData = new FormData();
                    const imageFile = document.getElementById('imageInput').files[0];
                    if (!imageFile) {
                        alert('Please select an image file');
                        return;
                    }
                    
                    formData.append('image', imageFile);
                    
                    document.getElementById('loading').style.display = 'block';
                    document.getElementById('result').innerHTML = '';
                    
                    try {
                        const response = await fetch('/outpaint', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (response.ok) {
                            const blob = await response.blob();
                            const url = URL.createObjectURL(blob);
                            
                            document.getElementById('result').innerHTML = `
                                <h3>Outpainted Image:</h3>
                                <img src="${url}" alt="Outpainted result">
                                <p><a href="${url}" download="outpainted-image.webp">Download Image</a></p>
                            `;
                        } else {
                            const error = await response.json();
                            document.getElementById('result').innerHTML = `<p>Error: ${error.error}</p>`;
                        }
                    } catch (error) {
                        document.getElementById('result').innerHTML = `<p>Error: ${error.message}</p>`;
                    } finally {
                        document.getElementById('loading').style.display = 'none';
                    }
                });
            </script>
        </body>
        </html>
        ''')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/outpaint', methods=['POST'])
def outpaint():
    input_tmp = None
    output_tmp = None
    temp_dir = None

    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image = request.files['image']
        if image.filename == '':
            return jsonify({"error": "No image selected"}), 400

        temp_dir = tempfile.mkdtemp()
        logging.debug(f"Created temporary directory: {temp_dir}")

        input_tmp = os.path.join(temp_dir, "input.png")
        image.save(input_tmp)
        logging.debug(f"Saved input image to: {input_tmp}")

        try:
            # Call the new /inpaint API with required parameters
            result = client.predict(
                image=handle_file(input_tmp),
                width=720,
                height=1280,
                overlap_percentage=10,
                num_inference_steps=28,
                resize_option="75%",
                custom_resize_percentage=50,
                prompt_input="",
                alignment="Bottom",
                overlap_left=True,
                overlap_right=True,
                overlap_top=True,
                overlap_bottom=True,
                api_name="/inpaint"
            )

            logging.debug(f"Result type: {type(result)}")
            logging.debug(f"Result content: {result}")
        except Exception as e:
            logging.exception(f"Error in Gradio client: {str(e)}")
            return jsonify({"error": f"Error in image processing: {str(e)}"}), 500

        # The result is a dict with a "path" key
        outpaint_result = None
        if isinstance(result, dict) and "path" in result:
            outpaint_result = result["path"]
        elif isinstance(result, list) and result and isinstance(result[0], dict) and "path" in result[0]:
            outpaint_result = result[0]["path"]
        elif isinstance(result, str):
            outpaint_result = result
        else:
            raise ValueError(f"Unexpected result format: {result}")

        if not outpaint_result or not os.path.exists(outpaint_result):
            raise ValueError(f"Result file not found: {outpaint_result}")

        output_tmp = os.path.join(temp_dir, "output.webp")
        shutil.copy2(outpaint_result, output_tmp)
        logging.debug(f"Copied result to: {output_tmp}")

        if not os.path.exists(output_tmp):
            raise ValueError(f"Copied file not found: {output_tmp}")

        return send_file(
            output_tmp,
            mimetype='image/webp',
            as_attachment=True,
            download_name='outpainted-image.webp'
        )

    except Exception as e:
        logging.exception(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logging.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logging.warning(f"Failed to delete temporary directory: {str(e)}")
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)