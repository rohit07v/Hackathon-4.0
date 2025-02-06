from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from utils.image_processing import process_images, generate_heatmap, extract_text_from_image

app = Flask(__name__)

# Configure file upload settings
UPLOAD_FOLDER = './static/uploads'
ANNOTATED_FOLDER = './static/annotated_overlay'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANNOTATED_FOLDER'] = ANNOTATED_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Get the uploaded files
        wireframe_file = request.files.get('wireframe')
        live_site_file = request.files.get('live_site')
        
        # Ensure files are valid
        if wireframe_file and allowed_file(wireframe_file.filename) and live_site_file and allowed_file(live_site_file.filename):
            wireframe_path = os.path.join(app.config['UPLOAD_FOLDER'], 'wireframe_image.png')
            live_site_path = os.path.join(app.config['UPLOAD_FOLDER'], 'live_site_image.png')
            
            # Save files
            wireframe_file.save(wireframe_path)
            live_site_file.save(live_site_path)

            # Process the images
            annotated_image_path, summary = process_images(wireframe_path, live_site_path, app.config['ANNOTATED_FOLDER'])
            
            # Generate heatmap
            try:
                ssim_score, heatmap_image_path = generate_heatmap(wireframe_path, live_site_path, annotated_image_path)
                heatmap_image_filename = os.path.basename(heatmap_image_path)
            except Exception as e:
                print(f"Error generating heatmap: {e}")
                heatmap_image_filename = None

            # Extract text from both images
            wireframe_text = extract_text_from_image(wireframe_path)
            live_site_text = extract_text_from_image(live_site_path)

            # Font/style comparison (basic for now, but can be expanded with better logic)
            font_style_mismatch = wireframe_text != live_site_text

            # Return results
            return render_template(
                "results.html", 
                wireframe_text=wireframe_text, 
                live_site_text=live_site_text, 
                summary=summary, 
                ssim_score=ssim_score,
                heatmap_image_filename=heatmap_image_filename,
                font_style_mismatch=font_style_mismatch
            )
        else:
            return jsonify({"error": "Invalid file type. Only image files are allowed."}), 400
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
