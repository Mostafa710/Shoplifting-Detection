from django import forms

class VideoUploadForm(forms.Form):
    video = forms.FileField(label="Upload a video (.mp4)", widget=forms.ClearableFileInput(attrs={"accept": "video/mp4"}))
