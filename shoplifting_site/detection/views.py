from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import subprocess
import os

from .forms import VideoUploadForm
from .inference import load_finetuned_model, predict_video

# Path where you will place the trained weights file copied from your notebook output
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best_3D_CNN_model_original_data.pth")

# Load model once at server start (CPU by default)
try:
    MODEL = load_finetuned_model(MODEL_PATH, device="cpu")
except Exception as e:
    MODEL = None
    MODEL_LOAD_ERROR = str(e)
else:
    MODEL_LOAD_ERROR = None

def home(request):
    context = { "form": VideoUploadForm() }
    if MODEL is None:
        context["error"] = "Model failed to load. Please ensure the weights file exists at: " + MODEL_PATH + "\nError: " + str(MODEL_LOAD_ERROR)
        return render(request, "index.html", context)

    if request.method == "POST":
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_file = form.cleaned_data["video"]
            fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, "uploads"))
            if not os.path.exists(fs.location):
                os.makedirs(fs.location, exist_ok=True)
            filename = fs.save(video_file.name, video_file)
            video_path = fs.path(filename)

            # ✅ Use the original video for inference
            result = predict_video(MODEL, video_path, device="cpu")

            # 🔄 Make a converted copy for web playback
            converted_path = os.path.splitext(video_path)[0] + "_converted.mp4"
            command = [
                "ffmpeg", "-y", "-i", video_path,
                "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                "-c:a", "aac", "-b:a", "128k",
                converted_path
            ]
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # URL for the converted video
            converted_url = os.path.join("uploads", os.path.basename(converted_path))
            context.update({ "result": result, "video_url": settings.MEDIA_URL + converted_url })
            return render(request, "result.html", context)
        else:
            context["form"] = form
    return render(request, "index.html", context)

def about(request):
    context = {
        "project_name": "Shoplifting AI",
        "authors": [
            {"name": "Mostafa Mamdouh", "role": "AI/ML Engineer"},
            {"name": "Abdullah Ashraf", "role": "AI/ML Enginner"},
        ],
        "architecture": "3D Convolutional Neural Network (3D-CNN)",
        "description": (
            "Shoplifting AI is a deep learning-based system designed to detect "
            "shoplifting activities from surveillance videos. "
            "The model leverages temporal and spatial features using a 3D-CNN "
            "architecture to achieve accurate detection in real-time scenarios."
        ),
        "contact_email": "mostafamamdouh710@gmail.com",
        "github": "https://github.com/Mostafa710",  # replace with real link
        "linkedin": "https://www.linkedin.com/in/mostafa-mamdouh-80b110228",  # example
        "year": "2025",
    }
    return render(request, "about.html", context)